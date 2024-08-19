import os
import sys
import time
import random
import math
import warnings
import statistics
import itertools
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.fft as fft

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Custom imports
import parameters as params
from data_loader import get_loader_segment, Dataset_classification
from augmentations import DataTransform_TD, DataTransform_FD

# Models
from models.anonymization_model import AnonymizationModel
from models.privacy_model import TFC
from models.loss import *
from models.anomaly_detection_model import AnomalyTransformer
from TimeSeriesProject.models.model.transformer import Privacy_Classifier

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Enable cuDNN benchmark mode for improved performance
torch.backends.cudnn.benchmark = True


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def _build_model_fa(device):
    model =  AnonymizationModel(
        data_dim=params.data_dim,
        in_len=params.in_len,
        seg_len=params.seg_len_fa,
        n_classes=params.n_classes,
        win_size=1,
        factor=params.factor_fa,
        d_model=params.d_model_fa,
        d_ff=params.d_ff_fa,
        n_heads=params.n_heads_fa,
        e_layers=params.e_layers_fa,
        dropout=params.dropout_fa,
        device=device
    ).float().to(device)

    return model


def _build_model_fb(device):
    model = TFC(
        data_dim=params.data_dim,
        in_len=params.in_len,
        d_model=params.d_model_fa,
        dim_feedforward=params.d_ff_fa,
        nhead=params.n_heads_fa,
        dropout=params.dropout_fa,
        num_layers=params.e_layers_fa,
    ).float().to(device)

    return model


def _build_model_ft(device):
    model = AnomalyTransformer(
        win_size=params.in_len,
        enc_in=params.data_dim,
        c_out=params.data_dim,
        d_model=params.d_model,
        n_heads=params.n_heads,
        e_layers=params.e_layers,
        d_ff=params.d_ff,
        dropout=params.dropout,
    ).float().to(device)

    return model


def _build_model_fc(device):
    model = Privacy_Classifier(
        d_model=params.d_model,
        n_head=params.n_heads,
        max_len=params.in_len,
        seq_len=params.in_len,
        ffn_hidden=params.d_model,
        n_layers=params.e_layers,
        drop_prob=params.dropout,
        details=False,
        n_classes=params.n_classes,
        in_features=params.data_dim,
        device=device
    ).float().to(device)

    return model


def _get_data(device, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = params.batch_size
    else:
        shuffle_flag = True
        batch_size = params.batch_size

    data_set = get_loader_segment(
        data=params.data,
        root_path=params.root_path,
        step_size=params.step_size,
        device=device,
        flag=flag,
        in_len=params.in_len,
        data_split=params.data_split,
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0
    )

    return data_set, data_loader


def train_epoch(epoch, data_loader, fa_model, fb_model, ft_model, optimizer_fa, optimizer_fb, optimizer_ft, learning_rate_fa, learning_rate_fb, learning_rate_ft, device):
    losses_fa, losses_fb, losses_ft = [], [], []
    criterion = nn.MSELoss()
    k = 3

    step = 1
    for i, (inputs, id)  in enumerate(data_loader):        
        if inputs.shape[0] != params.batch_size:
             continue

        inputs = inputs.to(torch.float32).to(device)

        # Optimizers reset
        optimizer_fa.zero_grad()
        optimizer_fb.zero_grad()
        optimizer_ft.zero_grad()

        # Training phase for fa model
        if step == 1:
            fa_model.train()
            fb_model.eval()
            ft_model.eval()

            inputs_with_id_embedding = inputs + fa_model.module.embedding_sensitive_info[id]

            aug = DataTransform_TD(inputs_with_id_embedding.detach().cpu().numpy())
            data_f = fft.fft(inputs_with_id_embedding.permute(0, 2, 1)).abs().permute(0, 2, 1)
            aug_f = DataTransform_FD(data_f, device)

            aug = torch.from_numpy(aug).to(torch.float32).to(device)

            # fb
            h_t, z_t, h_f, z_f = fb_model(fa_model(inputs_with_id_embedding), fa_model(data_f))
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = fb_model(fa_model(aug), fa_model(aug_f))

            # NTXentLoss: normalized temperature-scaled cross entropy loss
            nt_xent_criterion = NTXentLoss_poly(device, params.batch_size, params.temperature, params.use_cosine_similarity)
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            loss_tf = nt_xent_criterion(z_t, z_f)

            lam = 0.1
            loss_fb = loss_tf + lam * (loss_t + loss_f)

            # ft
            fa_input = fa_model(inputs_with_id_embedding)
            output, series, prior, _ = ft_model(fa_input)

            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)).detach())) 
                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)).detach(),
                    series[u])))
                
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)),
                    series[u].detach())) 
                    + torch.mean(my_kl_loss(series[u].detach(), (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)))))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output, fa_input)

            loss1 = rec_loss - k * series_loss
            loss2 = rec_loss + k * prior_loss
            loss_ft = loss1 + loss2

            loss_fa = -loss_fb + params.ft_loss_weight * loss_ft
            losses_fa.append(loss_fa.item())

            loss_fa.backward()
            optimizer_fa.step()
            step = 2

            if i % 100 == 0:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fa: {np.mean(losses_fa):.4f}')

        # Training phase for fb model and ft model
        elif step == 2:

            fa_model.eval()
            fb_model.train()
            ft_model.train()

            inputs_with_id_embedding = inputs + fa_model.module.embedding_sensitive_info[id]

            aug = DataTransform_TD(inputs_with_id_embedding.detach().cpu().numpy())
            data_f = fft.fft(inputs_with_id_embedding.permute(0, 2, 1)).abs().permute(0, 2, 1)
            aug_f = DataTransform_FD(data_f, device)

            aug = torch.from_numpy(aug).to(torch.float32).to(device)

            # fb
            h_t, z_t, h_f, z_f = fb_model(fa_model(inputs_with_id_embedding), fa_model(data_f))
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = fb_model(fa_model(aug), fa_model(aug_f))

            # NTXentLoss: normalized temperature-scaled cross entropy loss
            nt_xent_criterion = NTXentLoss_poly(device, params.batch_size, params.temperature, params.use_cosine_similarity)
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            loss_tf = nt_xent_criterion(z_t, z_f)

            lam = 0.1
            loss_fb = loss_tf + lam * (loss_t + loss_f)
            losses_fb.append(loss_fb.item())

            # ft
            inputs_with_id_embedding = inputs + fa_model.module.embedding_sensitive_info[id]
            fa_input = fa_model(inputs_with_id_embedding)
            output, series, prior, _ = ft_model(fa_input)

            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)).detach())) 
                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)).detach(),
                    series[u])))
                
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)),
                    series[u].detach())) 
                    + torch.mean(my_kl_loss(series[u].detach(), (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, params.win_size)))))
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output, fa_input)

            loss1 = rec_loss - k * series_loss
            loss2 = rec_loss + k * prior_loss
            loss_ft = loss1 + loss2
            losses_ft.append(loss_ft.item())

            # Minimax strategy
            loss1.backward(retain_graph=True)
            loss2.backward()
            loss_fb.backward()
            optimizer_ft.step()
            optimizer_fb.step()
            step = 1

            if i % 100 == 0:
                print(f'Training Epoch {epoch}, Batch {i}, loss_fb: {np.mean(losses_fb):.4f}, loss_ft: {np.mean(losses_ft):.4f}')

    return fa_model, fb_model, ft_model, np.mean(losses_fa), np.mean(losses_fb), np.mean(losses_ft)


def train_PP_TSAD(ii):
    save_dir = params.saved_models_dir
    
    # Create save directory if it does not exist.
    os.makedirs(save_dir, exist_ok=True)

    # Set the GPU
    if params.use_gpu:
        if params.use_multi_gpu:
            device_ids = [int(id_) for id_ in params.devices.replace(' ', '').split(',')]
            params.gpu = device_ids[0]
            device = torch.device(f'cuda:{params.gpu}')
            print(f'Use GPU: cuda:{params.gpu}')
        else:
            device = torch.device(f'cuda:{params.gpu}')
            print(f'Use GPU: cuda:{params.gpu}')
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # Build models
    fa_model = _build_model_fa(device)
    fb_model = _build_model_fb(device)
    ft_model = _build_model_ft(device)

    # Wrap models for multi-GPU usage
    if params.use_multi_gpu and params.use_gpu:
        fa_model = nn.DataParallel(fa_model, device_ids=device_ids)
        fb_model = nn.DataParallel(fb_model, device_ids=device_ids)
        ft_model = nn.DataParallel(ft_model, device_ids=device_ids)

    # Set learning rates and optimizers
    learning_rate_fa = params.learning_rate_fa
    learning_rate_fb = params.learning_rate_fb
    learning_rate_ft = params.learning_rate_ft

    optimizer_fa = optim.Adam(fa_model.parameters(), lr=params.learning_rate_fa)
    optimizer_fb = optim.Adam(fb_model.parameters(), lr=params.learning_rate_fb)
    optimizer_ft = optim.Adam(ft_model.parameters(), lr=params.learning_rate_ft)
    
    # Load data
    train_dataset, train_dataloader = _get_data(device, flag = 'train')
    val_dataset, val_dataloader = _get_data(device, flag = 'val')
    
    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset) / params.batch_size}')
    print(f'Validation dataset length: {len(val_dataset)}')
    print(f'Validation dataset steps per epoch: {len(val_dataset) / params.batch_size}')

    # Training loop
    for epoch in range(1, params.num_epochs + 1):    
        print(f'Epoch {epoch} started.')
        start = time.time()

        # Train models
        fa_model, fb_model, ft_model, fa_loss, fb_loss, ft_loss = train_epoch(
            epoch, train_dataloader, fa_model, fb_model, ft_model,
            optimizer_fa, optimizer_fb, optimizer_ft,
            learning_rate_fa, learning_rate_fb, learning_rate_ft,
            device
        )

        print(f"Train Epoch: {epoch} | Loss_ft: {ft_loss:.4f}")
        
        time_taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {time_taken}')
        print('*'*50)

        # Adjust learning rates
        adjust_learning_rate(optimizer_fa, epoch + 1, params.learning_rate_fa)
        adjust_learning_rate(optimizer_fb, epoch + 1, params.learning_rate_fb)
        adjust_learning_rate(optimizer_ft, epoch + 1, params.learning_rate_ft)
    
    # Save models
    file_name = f'checkpoint_{params.data}_itr{ii}.pth'
    save_file_path = os.path.join(save_dir, file_name)

    state_dict = {
        'fa_model_state_dict': fa_model.module.state_dict() if isinstance(fa_model, nn.DataParallel) else fa_model.state_dict(),
        'fb_model_state_dict': fb_model.module.state_dict() if isinstance(fb_model, nn.DataParallel) else fb_model.state_dict(),
        'ft_model_state_dict': ft_model.module.state_dict() if isinstance(ft_model, nn.DataParallel) else ft_model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state_dict, save_file_path)

    return fa_model, fb_model, ft_model, save_dir


def test_PP_TSAD(save_dir, fa_model, fb_model, ft_model):   
    # Set the device for computation
    device = torch.device(f'cuda:{params.gpu}' if params.use_gpu else 'cpu')
    print(f'Use GPU: cuda:{params.gpu}' if params.use_gpu else 'Use CPU')

    # Load data
    train_dataset, train_dataloader = _get_data(device, flag = 'train')
    test_dataset, test_dataloader = _get_data(device, flag = 'test')

    print(f'Test dataset length: {len(test_dataset)}')
    print(f'Test dataset steps per epoch: {len(test_dataset) / params.batch_size}')

    # Set models to evaluation mode
    fb_model.eval()
    ft_model.eval()
    
    # Define parameters and loss functions
    criterion = nn.MSELoss(reduce=False)
    temperature = 50

    start_time = time.time()
    attens_energy, train_collected_data, train_ids = [], [], []

    with torch.no_grad():
        # Stastic on the train set
        for i, (inputs, id) in enumerate(train_dataloader):
            inputs = inputs.float().to(device)
            inputs_with_id_embedding = inputs + fa_model.module.embedding_sensitive_info[id]
            fa_input = fa_model(inputs_with_id_embedding)
            output, series, prior, _ = ft_model(fa_input)
            loss = torch.mean(criterion(fa_input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                normed_prior = (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                                .repeat(1, 1, 1, params.win_size)).detach()

                if u == 0:
                    series_loss = my_kl_loss(series[u], normed_prior) * temperature
                    prior_loss = my_kl_loss(normed_prior, series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], normed_prior) * temperature
                    prior_loss += my_kl_loss(normed_prior, series[u].detach()) * temperature

            metric = torch.softmax(-series_loss - prior_loss, dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()

            attens_energy.append(cri)
            train_collected_data.append(fa_input.detach().cpu().numpy())
            train_ids.append(id)

        # Concatenate results
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        train_collected_data = np.concatenate(train_collected_data, axis=0)
        train_ids = np.concatenate(train_ids, axis=0).reshape(-1)

        # Evaluation on the test set
        attens_energy, test_labels, losses_fb, test_collected_data, test_ids = [], [], [], [], []
        anonymization_losses_test = []
        noise_L1loss = nn.L1Loss()

        for i, (inputs, label, id)  in enumerate(test_dataloader):
            inputs = inputs.to(torch.float32).to(device) 
            inputs_with_id_embedding = inputs + fa_model.module.embedding_sensitive_info[id]

            aug = DataTransform_TD(inputs_with_id_embedding.detach().cpu().numpy())
            data_f = fft.fft(inputs_with_id_embedding.permute(0, 2, 1)).abs().permute(0, 2, 1)
            aug_f = DataTransform_FD(data_f, device)

            aug = torch.from_numpy(aug).to(torch.float32).to(device)

            fa_input = fa_model(inputs_with_id_embedding)
            output, series, prior, _ = ft_model(fa_input)

            loss = torch.mean(criterion(fa_input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0

            for u in range(len(prior)):
                normed_prior = (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                                .repeat(1, 1, 1, params.win_size)).detach()

                if u == 0:
                    series_loss = my_kl_loss(series[u], normed_prior) * temperature
                    prior_loss = my_kl_loss(normed_prior, series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], normed_prior) * temperature
                    prior_loss += my_kl_loss(normed_prior, series[u].detach()) * temperature

            metric = torch.softmax(-series_loss - prior_loss, dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()

            attens_energy.append(cri)
            test_labels.append(label)
            test_collected_data.append(fa_input.detach().cpu().numpy())
            test_ids.append(id)

            # fb
            h_t, z_t, h_f, z_f = fb_model(inputs_with_id_embedding, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = fb_model(aug, aug_f)

            nt_xent_criterion = NTXentLoss_poly(device, params.batch_size, params.temperature, params.use_cosine_similarity)
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            loss_tf = nt_xent_criterion(z_t, z_f)
            lam = 0.1
            loss_fb = loss_tf + lam * (loss_t + loss_f)
            losses_fb.append(loss_fb.item())

            # Calculate anonymization loss
            loss_noise = noise_L1loss(inputs_with_id_embedding, fa_input)
            anonymization_losses_test.append(loss_noise.item())

        # Concatenate results
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        final_as = np.array(attens_energy)
        label = np.array(test_labels)
        combined_energy = np.concatenate([train_energy, final_as], axis=0)
        test_collected_data = np.concatenate(test_collected_data, axis=0)
        test_ids = np.concatenate(test_ids, axis=0).reshape(-1)

    # Point Adjusted Evaluation
    print('##### Point Adjusted Evaluation #####')
    thresh = np.percentile(combined_energy, 100 - params.anormly_ratio)
    pred = (final_as > thresh).astype(int)
    gt = label.astype(int)
    
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))

    # Window-wise Evaluation
    print('##### Window-wise Evaluation #####')
    thresholds = _simulate_thresholds(final_as, 1000)
    final_as_seq = _create_sequences(final_as, params.in_len, params.in_len)
    label_as_seq = _create_sequences(label, params.in_len, params.in_len)

    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    for th in tqdm(thresholds): # For each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(final_as_seq)): # For each sequence

            # if any part of the segment has an anomaly, we consider it as anomalous sequence
            true_anomalies, pred_anomalies = set(np.where(label_as_seq[t] == 1)[0]), set(np.where(final_as_seq[t] > th)[0])

            if len(pred_anomalies) > 0 and len(pred_anomalies.intersection(true_anomalies)) > 0:
                # Correct prediction (at least partial overlap with true anomalies)
                TP_t = TP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) == 0:
                # Correct rejection, no predicted anomaly on no true labels
                TN_t = TN_t + 1
            elif len(pred_anomalies) > 0 and len(true_anomalies) == 0:
                # False alarm (i.e., predict anomalies on no true labels)
                FP_t = FP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) > 0:
                # Predict no anomaly when there is at least one true anomaly within the sequence
                FN_t = FN_t + 1

        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)

    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 1e-7))
        recall.append(TP[i] / (TP[i] + FN[i] + 1e-7)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 1e-7))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-7))

    highest_th_idx = np.argmax(f1)
    print(f'Threshold: {thresholds[highest_th_idx]}')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            precision[highest_th_idx], recall[highest_th_idx], f1[highest_th_idx]))

    return (
        precision[highest_th_idx],
        recall[highest_th_idx],
        f1[highest_th_idx],
        np.mean(losses_fb),
        np.mean(anonymization_losses_test),
        np.squeeze(train_collected_data),
        np.squeeze(test_collected_data),
        train_ids,
        test_ids
    )


def _create_sequences(values, seq_length, stride, historical=False):
    seq = []

    if historical:
        for i in range(seq_length, len(values) + 1, stride):
            seq.append(values[i-seq_length:i])
    else:
        for i in range(0, len(values) - seq_length + 1, stride):
            seq.append(values[i : i + seq_length])

    return np.stack(seq)


def _simulate_thresholds(rec_errors, n):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds = []
    step_size = abs(np.max(rec_errors) - np.min(rec_errors)) / n
    th = np.min(rec_errors)
    thresholds.append(th)

    print(f'Threshold Range: ({np.min(rec_errors)}, {np.max(rec_errors)}) with Step Size: {step_size}')
    for i in range(n):
        th = th + step_size
        thresholds.append(th)

    return thresholds


#######################
# Classification
#######################

def calc_loss_and_score(pred, target, metrics):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    target = target.to(torch.int64)
    ce_loss = criterion(pred, target)

    metrics['loss'].append(ce_loss.item())
    pred = softmax(pred)
    _, pred = torch.max(pred, dim=1)
    correct = torch.sum(pred == target).item()
    metrics['correct'] += correct
    total = target.size(0)
    metrics['total'] += total

    return ce_loss


def print_average(metrics):
    loss = metrics['loss']
    average_loss = np.mean(loss)
    average_correct = (100 * metrics['correct']) / metrics['total']
    print(f'Average Loss: {average_loss:0.4f}  Average Correct: {average_correct:0.4f}')


def train_classification_epoch(epoch, data_loader, fc_model, optimizer_fc, learning_rate_fc, device):
    losses_fc = []

    metrics = {
        'loss': [],
        'correct': 0,
        'total': 0
    }

    for i, (inputs, label) in enumerate(data_loader):
        inputs = inputs.to(torch.float32).to(device)
        label = label.to(device)

        optimizer_fc.zero_grad()
        fc_model.train()

        output, _ = fc_model(inputs)
        loss_fc = calc_loss_and_score(output, label, metrics)
        loss_fc.backward()
        optimizer_fc.step()
        losses_fc.append(loss_fc.item())

        if i % 100 == 0:
            print('Training Epoch {0}, Batch {1}, Loss_fc: {2:.4f}'.format(epoch, i, np.mean(losses_fc)))

    return fc_model, np.mean(losses_fc)


def val_classification_epoch(data_loader, fc_model, device):
    fc_model.eval()
    losses_fc = []

    metrics = {
        'loss': [],
        'correct': 0,
        'total': 0
    }

    with torch.no_grad():
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)

            output, _ = fc_model(inputs)
            loss_fc = calc_loss_and_score(output, target, metrics)
            losses_fc.append(loss_fc.item())

        average_loss = np.mean(losses_fc)
        print(f'Validation Loss: {average_loss:.4f}')

    return average_loss


def train_Classification(train_dataset, train_id, save_dir, ii):
    if params.use_gpu:
        device = torch.device(f'cuda:{params.gpu}' if params.use_multi_gpu else 'cuda')
        if params.use_multi_gpu:
            device_ids = list(map(int, params.devices.replace(' ', '').split(',')))
            params.gpu = device_ids[0]
            print(f'Use GPUs: {device_ids}')
        else:
            print(f'Use GPU: cuda:{params.gpu}')
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # Initialize model and move to device
    fc_model = _build_model_fc(device)
    if params.use_multi_gpu and params.use_gpu:
        fc_model = nn.DataParallel(fc_model, device_ids=device_ids)

    # Set learning rates and optimizers
    learning_rate_fc = params.learning_rate_fc
    optimizer_fc = optim.Adam(fc_model.parameters(), lr=learning_rate_fc)

    # Split the dataset
    x_train, x_val, y_train, y_val = train_test_split(train_dataset, train_id, test_size = 0.2)
    train_dataset = Dataset_classification(x_train, y_train, params.data)
    val_dataset = Dataset_classification(x_val, y_val, params.data)

    print(f'Train dataset length: {len(train_dataset)}')
    print(f'Train dataset steps per epoch: {len(train_dataset) / params.batch_size:.2f}')
    print(f'Validation dataset length: {len(val_dataset)}')
    print(f'Validation dataset steps per epoch: {len(val_dataset) / params.batch_size:.2f}')

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Training model
    val_loss_min = np.Inf
    best_score = None
    counter = 0
    val_epochs = set(range(1, params.num_epochs + 1))

    for epoch in range(1, params.num_epochs + 1):
        print(f'Epoch {epoch} started.')
        start = time.time()
        early_stop = False

        fc_model, fc_loss = train_classification_epoch(epoch, train_dataloader, fc_model, optimizer_fc, learning_rate_fc, device)
        print(f'Train Epoch: {epoch} | Loss_fc: {fc_loss:.4f}')
        time_taken = time.time() - start
        print(f'Time taken for Epoch-{epoch} is {time_taken:.2f} seconds')

        # Validation
        if epoch in val_epochs:
            val_loss = val_classification_epoch(val_dataloader, fc_model, device)
            score = -val_loss
            print(f'Validation Epoch: {epoch} | Loss: {val_loss:.4f}')
            file_name = f'checkpoint_fc_{params.data}_itr{ii}.pth'

            if best_score is None:
                best_score = score
                print(f'Validation loss decreased ({val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...')

                save_file_path = os.path.join(save_dir, file_name)
                state_dict_fc = fc_model.module.state_dict() if isinstance(fc_model, nn.DataParallel) else fc_model.state_dict()

                torch.save({
                    'epoch': epoch,
                    'fc_model_state_dict': state_dict_fc,
                }, save_file_path)

                val_loss_min = val_loss
                best_fc_model = fc_model

            elif score < best_score:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {params.patience}')
                if counter >= params.patience:
                    early_stop = True

            else:
                best_score = score
                print(f'Validation loss decreased ({val_loss_min:.4f} --> {val_loss:.4f}). Saving model ...')

                save_file_path = os.path.join(save_dir, file_name)
                state_dict_fc = fc_model.module.state_dict() if isinstance(fc_model, nn.DataParallel) else fc_model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'fc_model_state_dict': state_dict_fc,
                }, save_file_path)
                val_loss_min = val_loss
                best_fc_model = fc_model
                counter = 0

        print('*'*50)

        if early_stop:
            print("Early stopping")
            break

        # # Adjust learning rate
        adjust_learning_rate(optimizer_fc, epoch + 1, params.learning_rate_fc)

    return fc_model


def test_Classification(save_dir, fc_model, test_dataset, test_id):
    if params.use_gpu:
        device = torch.device('cuda:{}'.format(params.gpu))
        print('Use GPU: cuda:{}'.format(params.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')

    # Prepare the test dataset and dataloader
    test_dataset = Dataset_classification(test_dataset, test_id, params.data)
    print('Test dataset length: {}'.format(len(test_dataset)))
    print('Test dataset steps per epoch: {}'.format(len(test_dataset)/params.batch_size))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Switch model to evaluation mode
    fc_model.eval()
    losses_fc = []
    metrics = {
        'loss': [],
        'correct': 0,
        'total': 0
    }

    # Evaluate the model
    with torch.no_grad():
        for i, (inputs, target) in enumerate(test_dataloader):
            inputs = inputs.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)

            output, enc_output = fc_model(inputs)
            calc_loss_and_score(output, target, metrics)
    
    print_average(metrics)

    return (100 * metrics['correct']) / metrics['total']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to train baseline')
    parser.add_argument("--saved_model", dest='saved_model', type=str, required=False, default=None)
    parser.add_argument("--itr", type=int, default=3, help='experiments times')

    args = parser.parse_args()
    print(f'Data: {params.data}, In_len: {params.in_len}')

    # Lists to store metrics
    precisions = []; recalls = []; f_scores = []
    losses_fb = []; anonymization_losses = []
    accuracy_list = []

    # Iterate over the number of experiments
    for ii in range(args.itr):
        print(f'{ii + 1} / {args.itr} times')

        # Anomaly Detection
        print('##### Anomaly Detection #####')
        fa_model, fb_model, ft_model, save_dir = train_PP_TSAD(ii)
        precision, recall, f_score, loss_fb, anonymization_loss, train_dataset, test_dataset, train_id, test_id = test_PP_TSAD(save_dir, fa_model, fb_model, ft_model)
        
        # Classification
        print('##### Classification #####')
        fc_model = train_Classification(train_dataset, train_id, save_dir, ii)
        accuracy = test_Classification(save_dir, fc_model, test_dataset, test_id)
        
        # Append metrics
        precisions.append(precision)
        recalls.append(recall)
        f_scores.append(f_score)
        losses_fb.append(loss_fb)
        anonymization_losses.append(anonymization_loss)
        accuracy_list.append(accuracy)

    # Summary of results
    print('##### Summary #####')
    print('***** Anomaly Detection *****')
    print(f'Average|| Pre: {statistics.mean(precisions):.4f}, Rec: {statistics.mean(recalls):.4f}, F1: {statistics.mean(f_scores):.4f}, Privacy Target Loss: {statistics.mean(losses_fb):.4f}, Anonymization Loss: {statistics.mean(anonymization_losses):.4f}')
    print(f'Standard Deviation|| Pre: {statistics.stdev(precisions):.4f}, Rec: {statistics.stdev(recalls):.4f}, F1: {statistics.stdev(f_scores):.4f}, Privacy Target Loss: {statistics.stdev(losses_fb):.4f}, Anonymization Loss: {statistics.stdev(anonymization_losses):.4f}')
    print('***** Classification *****')
    print(f'Average|| Acc: {statistics.mean(accuracy_list):.4f}')
    print(f'Standard Deviation|| Acc: {statistics.stdev(accuracy_list):.4f}')

