import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import parameters as params

warnings.filterwarnings('ignore')


class ECGSegLoader(Dataset):
    def __init__(self, data_name, root_path, step_size, device, flag='train', in_len=None, data_split = [0.8, 0.2], scale=True):
        self.in_len = in_len
        self.step_size = step_size
        self.device = device
        self.flag = flag
        self.data_split = data_split
        self.scale = scale

        self.data_name = data_name
        self.root_path = root_path

        self._read_data()

    def _read_data(self):
        data_folder = os.path.join(self.root_path, self.data_name)
        # Load the data
        train_data = np.load(os.path.join(data_folder, 'train.npy')).squeeze()
        test_data = np.load(os.path.join(data_folder, 'test.npy')).squeeze()
        test_labels = np.load(os.path.join(data_folder, 'test_label.npy')).squeeze()
        train_ids = np.load(os.path.join(data_folder, 'train_subject_id.npy'), allow_pickle=True).reshape(-1)
        test_ids = np.load(os.path.join(data_folder, 'test_subject_id.npy'), allow_pickle=True).reshape(-1)

        df_raw = pd.DataFrame(train_data).values
        df_raw_test = pd.DataFrame(test_data).values
        df_raw_test_label = pd.DataFrame(test_labels).values
        df_raw_train_id = pd.DataFrame(train_ids).values
        df_raw_test_id = pd.DataFrame(test_ids).values

        if self.flag == 'train':
            self.data_x = df_raw
            self.data_x = self.data_x[:, :, np.newaxis]  # Add channel dimension
            self.id = df_raw_train_id.reshape(-1)
        else:  # test
            self.data_x = df_raw_test
            self.data_x = self.data_x[:, :, np.newaxis]  # Add channel dimension
            self.test_label = df_raw_test_label
            self.id = df_raw_test_id.reshape(-1)

        print(f"Min ID: {min(self.id)}, Max ID: {max(self.id)}")

    def __getitem__(self, index):
        if self.flag == 'train':
            return self.data_x[index], self.id[index]
        else:  # test
            return self.data_x[index], self.test_label[index], self.id[index]

    def __len__(self):
        return len(self.data_x)


class TODSSegLoader(Dataset):
    def __init__(self, data_name, root_path, step_size, device, flag='train', in_len=None, data_split = [0.8, 0.2], scale=True):
        self.in_len = in_len
        self.step_size = step_size
        self.device = device
        self.flag = flag
        self.data_split = data_split
        self.scale = scale

        self.data_name = data_name
        self.root_path = root_path

        self._read_data()

    def _read_data(self):
        data_folder = os.path.join(self.root_path, self.data_name)
        # Load the data
        train_df = pd.read_csv(os.path.join(data_folder, 'train.csv')).values
        test_df = pd.read_csv(os.path.join(data_folder, 'test.csv')).values

        train_id = train_df[:, -1].astype(int)
        train_data = train_df[:, :5].astype(float)

        test_id = test_df[:, -1].astype(int)
        test_data = test_df[:, :5].astype(float)
        test_labels = test_df[:, -2].astype(int)

        if self.scale:
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_data)
            test_data = scaler.transform(test_data)

        if self.flag == 'train':
            self.data_x = train_data
            self.id = train_id
        else:  # test
            self.data_x = test_data
            self.test_label = test_labels
            self.id = test_id

    def __getitem__(self, index):
        index = index * self.step_size
        s_begin = index
        s_end = s_begin + self.in_len

        if self.flag == 'train':
            return self.data_x[s_begin:s_end], self.id[s_begin]
        else:  # test
            return self.data_x[s_begin:s_end], self.test_label[s_begin:s_end], self.id[s_begin]

    def __len__(self):
        return (len(self.data_x) - self.in_len + self.step_size) // self.step_size


class SWaTSegLoader(Dataset):
    def __init__(self, data_name, root_path, step_size, device, flag='train', in_len=None, data_split = [0.8, 0.2], scale=True):
        self.in_len = in_len
        self.step_size = step_size
        self.device = device
        self.flag = flag
        self.data_split = data_split
        self.scale = scale

        self.data_name = data_name
        self.root_path = root_path

        self._read_data()

    def _create_sequences(self, values, seq_length, stride, historical=False):
        sequences = []
        if historical:
            for i in range(seq_length, len(values) + 1, stride):
                sequences.append(values[i-seq_length:i])
        else:
            for i in range(0, len(values) - seq_length + 1, stride):
                sequences.append(values[i:i + seq_length])
        return np.stack(sequences)

    def _read_data(self):
        data_folder = os.path.join(self.root_path, self.data_name)
        # Load the data
        train_file = 'swat_train2.csv'
        test_file = 'swat2.csv'

        df_train = pd.read_csv(os.path.join(data_folder, train_file)).values[:, :-1]
        df_test = pd.read_csv(os.path.join(data_folder, test_file))
        test_labels = df_test.values[:, -1:]
        df_test = df_test.values[:, :-1]

        if self.scale:
            scaler = StandardScaler()
            df_train = scaler.fit_transform(df_train)
            df_test = scaler.transform(df_test)

        # Create sequences
        train_sequences = self._create_sequences(df_train, self.in_len, self.step_size)
        test_sequences = self._create_sequences(df_test, self.in_len, self.step_size)
        test_labels_sequences = self._create_sequences(test_labels, self.in_len, self.step_size)

        if self.flag == 'train':
            self.data_x = train_sequences
            self.id = np.random.randint(0, params.n_classes, size=len(self.data_x))
        else:  # 'test'
            self.data_x = test_sequences
            self.test_label = test_labels_sequences
            self.id = np.random.randint(0, params.n_classes, size=len(self.data_x))

    def __getitem__(self, index):
        if self.flag in ['train', 'val']:
            return self.data_x[index], self.id[index]
        else:
            return self.data_x[index], self.test_label[index], self.id[index]

    def __len__(self):
        return len(self.data_x)


class PSMSegLoader(Dataset):
    def __init__(self, data_name, root_path, step_size, device, flag='train', in_len=None, data_split = [0.8, 0.2], scale=True):
        self.in_len = in_len
        self.step_size = step_size
        self.device = device
        self.flag = flag
        self.data_split = data_split
        self.scale = scale

        self.data_name = data_name
        self.root_path = root_path

        self._read_data()

    def _create_sequences(self, values, seq_length, stride, historical=False):
        """
        Creates sequences from the input data.
        """
        sequences = []
        if historical:
            for i in range(seq_length, len(values) + 1, stride):
                sequences.append(values[i-seq_length:i])
        else:
            for i in range(0, len(values) - seq_length + 1, stride):
                sequences.append(values[i : i + seq_length])

        return np.stack(sequences)

    def _read_data(self):
        data_folder = os.path.join(self.root_path, self.data_name)
        # Load the data
        train_data = pd.read_csv(os.path.join(data_folder, 'train.csv')).values[:, 1:]
        test_data = pd.read_csv(os.path.join(data_folder, 'test.csv')).values[:, 1:]
        test_labels = pd.read_csv(os.path.join(data_folder, 'test_label.csv')).values[:, 1:]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

        df_raw_test_label = pd.read_csv(os.path.join(data_folder, 'test_label.csv'))
        df_raw_test_label = df_raw_test_label.values[:, 1:]

        # Create sequences
        train_sequences = self._create_sequences(train_data, self.in_len, self.step_size)
        test_sequences = self._create_sequences(test_data, self.in_len, self.step_size)
        test_labels_sequences = self._create_sequences(df_raw_test_label, self.in_len, self.step_size)

        if self.flag == 'train':
            self.data_x = train_sequences
            self.id = np.random.randint(0, params.n_classes, size=len(self.data_x))
        else:  # 'test'
            self.data_x = test_sequences
            self.test_label = test_labels_sequences
            self.id = np.random.randint(0, params.n_classes, size=len(self.data_x))

    def __getitem__(self, index):
        if self.flag in ['train', 'val']:
            return self.data_x[index], self.id[index]
        else:
            return self.data_x[index], self.test_label[index], self.id[index]

    def __len__(self):
        return len(self.data_x)


class Dataset_classification(torch.utils.data.Dataset):
    def __init__(self, data, id, data_name):
        self.data = data
        self.id = id
        # Adjust the shape of data if necessary
        if data_name == 'ECG':
            self.data = self.data[:, :, np.newaxis]

    def __getitem__(self, index):
        return self.data[index], self.id[index]

    def __len__(self):
        return len(self.data)


def get_loader_segment(data, root_path, step_size, device, flag='train', in_len=None, data_split = [0.8, 0.2]):
    if (data == 'ECG'):
        dataset = ECGSegLoader(data, root_path, step_size, device, flag, in_len, data_split, scale=True)
    elif (data == 'TODS'):
        dataset = TODSSegLoader(data, root_path, step_size, device, flag, in_len, data_split, scale=True)
    elif (data == 'SWaT'):
        dataset = SWaTSegLoader(data, root_path, step_size, device, flag, in_len, data_split, scale=True)
    elif (data == 'PSM'):
        dataset = PSMSegLoader(data, root_path, step_size, device, flag, in_len, data_split, scale=True)
    else:
        raise ValueError(f"Unsupported data type: {data}")
    
    return dataset

