import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
from kan import *

ic=200
oc=300
ws=13
fs=(ws+1)//2
png_in=fs*fs

tc=oc
encoder_in=png_in
dim_f=encoder_in
L1O=128

num_class=16

def BasicConv(in_channels, out_channels, kernel_size, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),)

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class MyModel(torch.nn.Module):
    def __init__(self,d_model=oc*png_in,num_encoder_layers=3,nhead=7,dropout=0.1,dim_feedforward=dim_f,batch_size=32):
        super(MyModel, self).__init__()
        
        self.conv0 = BasicConv(in_channels=ic, out_channels=oc, kernel_size=3, stride=2, padding=1) 
        #Block
        self.conv1 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)

        self.conv3 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv4 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)

        self.conv5 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv6 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)

        #KAN
        self.linear1=KANLinear(encoder_in*oc, L1O)
        self.linear2=KANLinear(L1O, num_class)   

    def forward(self, value):
        batch_size=value.shape[0]

        value=self.conv0(value)
        
        identity=value
        value=self.conv1(value)
        value=self.conv2(value)+identity
        
        identity=value
        value=self.conv3(value)
        value=self.conv4(value)+identity  

        identity=value
        value=self.conv3(value)
        value=self.conv4(value)+identity
        
        #KAN
        value=value.reshape(batch_size,-1)

        value=self.linear1(value)
        value=self.linear2(value)

        return value