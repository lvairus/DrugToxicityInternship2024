# imports
# Load model directly
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from binary_nnmodel import NNModel
from data_utils import CustomDataset
import sys
import pdb
import yaml
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, SUPPRESS
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
import random
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch.backends.cudnn 
import torch.cuda
    
# ========================================================================================================================

# parse arguments: read in yaml file with all hyperparameters
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-y", "--yaml", type=Path, required=False, default="binary_config.yaml", help="path to config .yaml file"
)
# args = parser.parse_args()
args, unknown_args = parser.parse_known_args()
with open(args.yaml, 'r') as file:
    config_dict = yaml.safe_load(file)


# init wandb to log results
wandb.init( project = config_dict["init_project"],
            group = config_dict["init_group"],
            notes = "molform encs",
            config = config_dict,
)
config = wandb.config
cutoff = config.cutoff 

# ========================================================================================================================

# Reproducability
seeds = [53844, 837465, 800662, 910250, 543584, 179839, 707873, 482701, 278083, 198125]
SEED = seeds[config.seed_idx]

def set_determenistic_mode(SEED):
    torch.manual_seed(SEED)                         # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)                               # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))   # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)             
    torch.cuda.manual_seed_all(SEED)                # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_determenistic_mode(SEED)
gen = torch.Generator()
gen.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ========================================================================================================================

# initialize models
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
LLModel.to("cuda")
LLModel.eval()

nnmodel = NNModel(config).to("cuda")

wandb.watch(nnmodel, log_freq=100)


# ========================================================================================================================

# Data Preprocessing

# load in pre-split data 
train_df = pd.read_csv("data/mouse_train_df.csv")
test_df = pd.read_csv("data/mouse_test_df.csv")
cutoff_to_category = {10:0, 50:1, 500:2, 2000:3}
cutoff_idx = cutoff_to_category[cutoff]

# make binary column 
train_df["tox"] =  (train_df["EPACategoryIndex"] <= cutoff_idx).astype(int)
test_df["tox"] =  (test_df["EPACategoryIndex"] <= cutoff_idx).astype(int)

# get counts of positive and negative samples
num_pos = len(train_df[train_df["tox"] == 1])
num_neg = len(train_df[train_df["tox"] == 0])

# define train and test sets
X_train = train_df['SMILES']
X_test = test_df["SMILES"]
Y_train = train_df['tox']
Y_test = test_df["tox"]

# convert feature pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()
# convert label pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist(), dtype=torch.float)
Y_test = torch.tensor(Y_test.tolist(), dtype=torch.float)

# create CustomDataset object
training_dataset = CustomDataset(tokenizer, X_train, Y_train, max_input_length=512, max_target_length=512) 
test_dataset = CustomDataset(tokenizer, X_test, Y_test, max_input_length=512, max_target_length=512) 

# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)

taskname = config.model_type


# ========================================================================================================================

encodings = []
labels = []

for batch in train_dataloader:
    # pass through Molformer
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    y_regression_values = batch["y_regression_values"]
    with torch.no_grad():
        outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoder = outputs["hidden_states"][-1]
    # average over second dimension of encoder output to get a single vector for each example
    encoder = encoder.mean(dim=1)

    # save encs 
    encodings.extend(encoder.cpu().numpy().tolist())
    labels.extend(y_regression_values.cpu().numpy().tolist())


encoding_df = pd.DataFrame({'encodings': encodings, 'labels': labels})
encoding_df.to_csv(f"molform_encs/{taskname}_molform_encs.csv", index=False)
