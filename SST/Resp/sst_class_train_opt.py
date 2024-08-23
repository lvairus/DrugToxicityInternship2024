import os, sys
from torchsummary import summary
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
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import json
from sst_class_lv import *
from torcheval.metrics import R2Score
from torcheval.metrics import MulticlassAccuracy
from tqdm import tqdm
import wandb

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

def tokenize_function(examples, ntoken) :
    return np.array(list(pad(tokenizer(examples)['input_ids'], ntoken, 0)))

def training_data(raw_data):
    smiles_data_frame = pd.DataFrame(data = {'text': raw_data['Canonical SMILES'], 'labels': raw_data['Toxicity Value']})
    smiles_data_frame['text'] = smiles_data_frame['text'].apply(lambda x: tokenize_function(x, ntoken=ntoken))#map(tokenize_function)#, batched=True)
    target = smiles_data_frame['labels'].values
    features = np.stack([tok_dat for tok_dat in smiles_data_frame['text']])
    feature_tensor = torch.tensor(features)
    label_tensor = torch.tensor(smiles_data_frame['labels'])
    dataset = TensorDataset(feature_tensor, label_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = int(len(dataset) - train_size)

    training_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)
    return train_dataloader, test_dataloader, test_data


def dataload_presplit(traindat, valdat, smilescol, labelcol, batch, ntoken):
    tqdm.pandas()
    smiles_df_train = pd.DataFrame(data = {'text': traindat[smilescol], 'labels': traindat[labelcol]})
    smiles_df_val = pd.DataFrame(data = {'text': valdat[smilescol], 'labels': valdat[labelcol]})

    smiles_df_train['text'] = smiles_df_train['text'].progress_apply(lambda x: tokenize_function(x, ntoken=ntoken))
    target_train = smiles_df_train['labels'].values
    features_train = [tok_dat for tok_dat in smiles_df_train['text']]#np.stack([tok_dat for tok_dat in smiles_df_train['text']])
    smiles_df_val['text'] = smiles_df_val['text'].progress_apply(lambda x: tokenize_function(x, ntoken=ntoken))
    target_val = smiles_df_val['labels'].values
    features_val = [tok_dat for tok_dat in smiles_df_val['text']]#np.stack([tok_dat for tok_dat in smiles_df_val['text']])

    feature_tensor_train = torch.tensor(features_train)
    label_tensor_train = torch.tensor(smiles_df_train['labels'])
    feature_tensor_val = torch.tensor(features_val)
    label_tensor_val = torch.tensor(smiles_df_val['labels'])

    train_dataset = TensorDataset(feature_tensor_train, label_tensor_train)
    val_dataset = TensorDataset(feature_tensor_val, label_tensor_val)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    return train_dataloader, val_dataloader, val_dataset

#############################################################
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-t", "--traindata", type=Path, required=True, help="Input data for training"
)

parser.add_argument(
    "-v", "--valdata", type=Path, required=True, help="Input data for validation"
)

parser.add_argument(
    "-s", "--smilescol", type=str, required=True, help="Column for SMILES"
)

parser.add_argument(
    "-l", "--labelcol", type=str, required=True, help="Column for label"
)

parser.add_argument(
    "-c", "--confmod", type=Path, required=True, help="config file for model"
)

parser.add_argument(
    "-e", "--epochs", type=int, required=True, help="number of epochs"
)

parser.add_argument(
    "-b", "--batch", type=int, required=True, help="batch size"
)

parser.add_argument(
    "-L", "--lr", type=float, required=False, default=1e-5, help="batch size"
)

args = parser.parse_args()
with open(args.confmod, 'r') as f:
        config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Single task learning tests
'''
traindat = pd.read_csv(args.traindata)
valdat = pd.read_csv(args.valdata)
traindat['label_scaled'] = (traindat[args.labelcol] - traindat[args.labelcol].min()) / (traindat[args.labelcol].max() - traindat[args.labelcol].min())

valdat['label_scaled'] = (valdat[args.labelcol] - valdat[args.labelcol].min()) / (valdat[args.labelcol].max() - valdat[args.labelcol].min())


base = 'training'

train_loader, val_loader, val_data = dataload_presplit(traindat, valdat, args.smilescol, 'label_scaled', args.batch, config['ntoken'])

lr_list = [0.0001, 0.00001, 0.000001]
head_list = [16,32,64]
layer_list = [4,8,16]

dict_hyperparams = 
{   "nhead": [],
    "nlayers": [],
    "lr": [],
    "loss": [],
    "val_mca": [],
    "val_mcauroc": []}

for l in lr_list:
    
    for head in head_list:

        for layer in layer_list:

            dict_hyperparams["nhead"].append(head)
            dict_hyperparams["nlayers"].
            #...

            model = TransformerModel(ntoken=config['ntoken'], d_model=config['d_model'], nhead=head, d_hid=config['d_hid'],
                            nlayers=layer, dropout= config['dropout'], device=device)
            model = model.to(device)

            with torch.no_grad():
                
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                
                print(f"model size is {param_size}")

            # define optimizer. Specify the parameters of the model to be trainable. Learning rate of .001
            optimizer = torch.optim.Adam(model.parameters(), lr = l)
            loss_fn = nn.MSELoss()

            # some extra variables to track performance during training
            r2_history = []
            r2_vals = []
            trainstep = 0
            running_loss = 0.
            metric = R2Score().to(device)

            wandb.init()

            # Magic
            wandb.watch(model, log_freq=100)

            model.train()

            for i in tqdm(range(args.epochs)):
                for j, (batch_X, batch_y) in enumerate(train_loader):
                    preds = model(batch_X.to(device)).flatten()
                    loss = loss_fn(preds.flatten(), batch_y.float().to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    del(preds)
                    wandb.log({"loss": loss})
                y_hat_total = []
                y_grnd_total = []
                with torch.no_grad():
                    for k, (batch_Xt, batch_yt) in enumerate(val_loader):
                        y_hat = model(batch_Xt.to(device)).flatten()
                        y_grnd = batch_yt.float().to(device)
                        y_hat_total.extend(y_hat.cpu().detach().numpy())
                        y_grnd_total.extend(y_grnd.cpu().detach().numpy())
                    #print(y_hat_total)
                    #print(y_grnd_total)
                    #r2_k = sklearn.metrics.r2_score(y_grnd_total, y_hat_total)
                        metric.update(y_hat, y_grnd)
                        r2_k = metric.compute()
                        #r2_k = r2_k.cpu().detach().numpy()
                    r2_history.append({'epoch' : i, 'minibatch' : k, 'trainstep' : trainstep,
                                                'task' : 'tox', 'binacc' : r2_k})
                    r2_vals.append(r2_k.cpu().detach().numpy())
                    trainstep += 1
                    wandb.log({"val-r2": r2_k})
                    if r2_k.cpu().detach().numpy() == np.max(r2_vals):
                        torch.save({
                            'epoch': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, f'models/{base}.pt')

            dict_hyperparams["loss"].append(loss)
            dict_hyperparams["val_r2"].append(r2_k)
            r2_df = pd.DataFrame(r2_history)
            r2_df.to_csv(f'r2_{base}.csv')

df_hyperparams = pd.DataFrame.from_dict(dict_hyperparams)
df_hyperparams.to_csv("hyperparams_test.csv")
