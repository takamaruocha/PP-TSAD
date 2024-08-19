"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from TimeSeriesProject.models.embedding.positional_encoding import PositionalEncoding
from TimeSeriesProject.models.layers.classification_head import ClassificationHead

from TimeSeriesProject.models.model.encoder import Encoder
  

class Privacy_Classifier(nn.Module):

    def __init__(self, device, d_model=100, n_head=4, max_len=5000, seq_len=200,
                 ffn_hidden=128, n_layers=2, drop_prob=0.1, details =False, n_classes=5, in_features=1):
        super().__init__() 
        self.device = device
        self.details = details
        self.n_classes = n_classes
        self.encoder_input_layer = nn.Linear(   
            in_features=in_features, 
            out_features=d_model 
            )
   
        self.pos_emb = PositionalEncoding( max_seq_len=max_len,batch_first=False, d_model=d_model, dropout=0.1) 
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head, 
                               ffn_hidden=ffn_hidden, 
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               details=details,
                               device=device)
        self.classHead = ClassificationHead(seq_len=seq_len,d_model=d_model,details=details,n_classes=n_classes)

    def forward(self, src): 
        if self.details: print('before input layer: '+ str(src.size()) )
        src= self.encoder_input_layer(src)
        if self.details: print('after input layer: '+ str(src.size()) )
        src= self.pos_emb(src)
        if self.details: print('after pos_emb: '+ str(src.size()) )
        enc_src = self.encoder(src) 
        cls_res, enc_output = self.classHead(enc_src)
        if self.details: print('after cls_res: '+ str(cls_res.size()) )
        return cls_res, enc_output
