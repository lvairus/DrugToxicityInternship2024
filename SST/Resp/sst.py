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
import seaborn as sns
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
'''
Initialize tokenizer
'''
vocab_file = 'VocabFiles/vocab_spe.txt'
spe_file = 'VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def tokenize_function(examples):
    #print(examples[0])
    return np.array(list(pad(tokenizer(examples)['input_ids'], 25, 0)))

def training_data(raw_data):
    smiles_data_frame = pd.DataFrame(data = {'text': raw_data['Canonical SMILES'], 'labels': raw_data['Toxicity Value']})
    print(smiles_data_frame['text'])
    smiles_data_frame['text'] = smiles_data_frame['text'].map(tokenize_function)#, batched=True)
    print(smiles_data_frame['text'].values)
    target = smiles_data_frame['labels'].values
    features = np.stack([tok_dat for tok_dat in smiles_data_frame['text']])
    print(target)
    #train = data_utils.TensorDataset(features, target)
    #train_loader = data_utils.DataLoader(train, batch_size=10, shuffle=True)
    feature_tensor = torch.tensor(features)
    label_tensor = torch.tensor(smiles_data_frame['labels'])
    print(feature_tensor)
    print(label_tensor)
    dataset = TensorDataset(feature_tensor, label_tensor)
    print(len(dataset[0][0]))
    train_size = int(0.9 * len(dataset))
    test_size = int(len(dataset) - train_size)

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    #print(training_data.shape)
    print(len(test_data))
    test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=True)
    return train_dataloader, test_dataloader, test_data


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
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        
        self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        self.layer_norm = nn.LayerNorm(d_model) 
        self.embedding = nn.Embedding(3132, d_model)
        self.d_model = d_model

        self.dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(6400, 2048)
        self.act1 = nn.ReLU()

        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(2048, 1024)
        self.act2 = nn.ReLU()

        self.dropout3 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(1024, 256)
        self.act3 = nn.ReLU()

        self.dropout4 = nn.Dropout(0.2)
        self.linear4 = nn.Linear(256, 64)
        self.act4 = nn.Softmax()
        #self.act4 = torch.sigmoid()

        self.dropout5 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(64, 16)
        self.act5 = nn.Softmax()

        self.dropout6 = nn.Dropout(0.2)
        self.linear6 = nn.Linear(16, 1)
        self.act6 = nn.Softmax()



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
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder1(src, src_mask)
        output = self.transformer_encoder2(output)
        #output = self.layer_norm(output)
        output = self.dropout1(output)
        output = torch.reshape(output, (len(output),len(output[0])*len(output[0][0])))
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
        output = torch.sigmoid(output)
        output = self.dropout5(output)
        output = self.linear5(output)
        output = torch.sigmoid(output)
        output = self.dropout6(output)
        output = self.linear6(output)
        output = torch.sigmoid(output)
        #output = self.act5(output)
        return torch.reshape(output, (-1,))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = 'data'
dir_list = os.listdir(directory)

'''
Single task learning tests
'''
if True:
    for fil in dir_list:
        raw = pd.read_csv(f'{directory}/{fil}')
        #raw = pd.read_csv(f'data/Endocrine_Disruption_NR-AR.csv')
        base = Path(f'{fil}').stem
    
        train_loader, test_loader, test_data = training_data(raw)
        
        model = TransformerModel(ntoken=25, d_model=256, nhead=16, d_hid=256,
                         nlayers=8, dropout= 0.2)
        model = model.to(device) 
        
        # define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
        loss_fn = nn.BCELoss()#NLLLoss() #nn.
        
        # some extra variables to track performance during training
        f1_history = []
        f1_vals = []
        trainstep = 0
        running_loss = 0.
        metric = BinaryAccuracy()
        for i in tqdm(range(50)):
            for j, (batch_X, batch_y) in enumerate(train_loader):
                preds = model(batch_X.to(device))
                #print(preds)
                loss = loss_fn(preds, batch_y.float().to(device))
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            for k, (batch_Xt, batch_yt) in enumerate(test_loader):
                y_hat = model(batch_Xt.to(device))
                #print(y_hat)
                y_hat =y_hat#(torch.round(y_hat)).long()#int64()# >=.5
                #print(y_hat)
                y_grnd = batch_yt.long().to(device)#==1
                #print(y_grnd)
                metric.update(y_hat, y_grnd)
                acc_k = metric.compute()
                #f1_k = multiclass_f1_score(y_hat, y_grnd, num_classes=2, average="macro")
                f1_history.append({'epoch' : i, 'minibatch' : k, 'trainstep' : trainstep,
                                          'task' : 'tox', 'binacc' : acc_k})
                f1_vals.append(acc_k)
                trainstep += 1
            
            if acc_k == np.max(f1_vals):
                print(f1_history)
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f'models/{base}.pt')
    
        f1_df = pd.DataFrame(f1_history)
        f1_df.to_csv(f'f1_{base}.csv')

