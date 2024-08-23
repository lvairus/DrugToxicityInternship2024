import os, sys
import pandas as pd
import scipy as sp
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import movie_reviews
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader, dataset
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
from itertools import chain, repeat, islice
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics import BinaryAccuracy
warnings.filterwarnings("ignore")
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, device):
        super().__init__()
        self.device = device
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.ntoken = ntoken
        self.d_model = d_model
        self.dropout = dropout
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        
        self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        #self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.layer_norm = nn.LayerNorm(d_model) 
        self.embedding = nn.Embedding(3132, d_model)
        self.d_model = d_model

        self.dropout1 = nn.Dropout(self.dropout)
        self.linear1 = nn.Linear(self.d_model*self.ntoken, 1024)
        self.act1 = nn.GELU()

        self.dropout2 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(1024,512)
        self.act2 = nn.GELU()

        self.dropout3 = nn.Dropout(self.dropout)
        self.linear3 = nn.Linear(512,256)
        self.act3 = nn.GELU()

        self.dropout4 = nn.Dropout(self.dropout)
        self.linear4 = nn.Linear(256, 64)
        self.act4 = nn.GELU()
        #self.act4 = torch.sigmoid()

        self.dropout5 = nn.Dropout(self.dropout)
        self.linear5 = nn.Linear(64, 16)
        self.act5 = nn.GELU()

        self.dropout6 = nn.Dropout(0.2)
        self.linear6 = nn.Linear(16, 5)
        # self.act6 = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.bias.data.zero_()
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear5.bias.data.zero_()
        self.linear5.weight.data.uniform_(-initrange, initrange)
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src)* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if False:
            if src_mask is None:
                """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
                """
                src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder1(src)#, src_mask=None)
        #output = self.transformer_encoder2(output)
        #output = self.layer_norm(output)
        output = self.dropout1(output)
        output = torch.reshape(output, (len(output), self.d_model*self.ntoken))# (len(output),len(output[0])*len(output[0][0])))
        output = self.linear1(output)
        output = self.act1(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        output = self.act2(output)
        output = self.dropout3(output)
        output = self.linear3(output)
        output = self.act3(output)
        output = self.dropout4(output)
        output = self.linear4(output)
        #output = torch.sigmoid(output)
        output = self.act4(output)
        output = self.dropout5(output)
        output = self.linear5(output)
        output = self.act5(output)
        output = self.dropout6(output)
        output = self.linear6(output)
        #output = self.act6(output)
        #output = self.act5(output)
        return output#torch.reshape(output, (-1,))


