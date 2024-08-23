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
from data_utils_epacatidx import CustomDataset
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
    "-y", "--yaml", type=Path, required=False, default="config.yaml", help="path to config .yaml file"
)
# args = parser.parse_args()
args, unknown_args = parser.parse_known_args()
with open(args.yaml, 'r') as file:
    config_dict = yaml.safe_load(file)


# init wandb to log results
wandb.init( project = config_dict["init_project"],
            group = config_dict["init_group"],
            notes = config_dict["init_notes"],
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

# # import data
# filepath = config.dataset
# data = pd.read_csv(filepath)
# # split data
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
#     data[config.smilescol],
#     data[config.labelcol],
#     test_size=config.testprop,
#     shuffle=True,
#     stratify=data[config.labelcol],
#     random_state=SEED
# )
# # save split
# train_df = pd.DataFrame({"SMILES": X_train.values, "EPACategoryIndex": Y_train.values})
# test_df = pd.DataFrame({"SMILES": X_test.values, "EPACategoryIndex": Y_test.values})
# train_df.to_csv("mouse_train_df.csv", index=False)
# test_df.to_csv("mouse_test_df.csv", index=False)

# load in pre-split data 
train_df = pd.read_csv("data/mouse_train_df.csv")
test_df = pd.read_csv("data/mouse_test_df.csv")
cutoff_to_category = {10:0, 50:1, 500:2, 2000:3}
cutoff_idx = cutoff_to_category[cutoff]

# if tree model, filter data based on tree 
if config.model_type == "tree":
    if cutoff_idx == 0:
        train_df = train_df[train_df['EPACategoryIndex'] <= 1]
        test_df = test_df[test_df['EPACategoryIndex'] <= 1]
    elif cutoff_idx == 1:
        train_df = train_df[train_df['EPACategoryIndex'] <= 2]
        test_df = test_df[test_df['EPACategoryIndex'] <= 2]
    elif cutoff_idx == 3:
        train_df = train_df[train_df['EPACategoryIndex'] >= 3]
        test_df = test_df[test_df['EPACategoryIndex'] >= 3]

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
epacatidxs_train = train_df["EPACategoryIndex"]
epacatidxs_test = test_df["EPACategoryIndex"]

# convert feature pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()
# convert label pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist(), dtype=torch.float)
Y_test = torch.tensor(Y_test.tolist(), dtype=torch.float)
epacatidxs_train = torch.tensor(epacatidxs_train.tolist(), dtype=torch.int)
epacatidxs_test = torch.tensor(epacatidxs_test.tolist(), dtype=torch.int)

# create CustomDataset object
training_dataset = CustomDataset(tokenizer, X_train, Y_train, epacatidxs_train, max_input_length=512, max_target_length=512) 
test_dataset = CustomDataset(tokenizer, X_test, Y_test, epacatidxs_test, max_input_length=512, max_target_length=512) 

# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)

taskname = config.model_type


# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config.lr)
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

early_stop = 20
stop_crit = 0
best_ep_auc = 0
best_ep_prec = 0
best_ep_recall = 0
best_ep_f1 = 0
best_ep_acc = 0


if config.loss_fxn == "weighted":
    # Weighted Binary Cross-Entropy Loss
    pos_weight = torch.tensor([num_neg/num_pos], device="cuda")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    loss_fn = nn.BCELoss()


for epoch in tqdm(range(config.epochs)):
    wandb.log({'epoch': epoch})
    # training
    if True:
        train_running_loss = 0

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
            # pass through our model
            preds = nnmodel(encoder)         
            loss = loss_fn(preds.flatten(), y_regression_values)
            train_running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_avg_loss = train_running_loss / len(train_dataloader)
        wandb.log({'train loss': train_avg_loss})

    # validation
    if True:
        val_preds = []
        val_labels = []
        val_running_loss = 0

        for batch in test_dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            y_regression_values = batch["y_regression_values"]
            with torch.no_grad():
                outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                encoder = outputs["hidden_states"][-1]
                encoder = encoder.mean(dim=1)

                preds  = nnmodel(encoder)
                loss = loss_fn(preds.flatten(), y_regression_values)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_regression_values.cpu().numpy())

                val_running_loss += loss

        val_avg_loss = val_running_loss / len(test_dataloader)
        auc = roc_auc_score(val_labels, val_preds) # calc_auc

        binary_preds = (np.array(val_preds) > 0.5).astype(int)
        binary_labels = np.array(val_labels).astype(int)
        val_prec = precision_score(binary_labels, binary_preds, zero_division=0)
        val_recall = recall_score(binary_labels, binary_preds)
        val_f1 = f1_score(binary_labels, binary_preds)
        val_acc = accuracy_score(binary_labels, binary_preds)
        
        wandb.log({'val loss': val_avg_loss})
        wandb.log({'val auc': auc})
        wandb.log({'val precision': val_prec})
        wandb.log({'val recall': val_recall})
        wandb.log({'val f1': val_f1})


        # early stopping and saving best results
        if val_f1>best_ep_f1:
            stop_crit = 0
            best_ep = epoch
            best_ep_tloss = train_avg_loss
            best_ep_vloss = val_avg_loss
            best_ep_auc = auc
            best_ep_prec = val_prec
            best_ep_recall = val_recall
            best_ep_f1 = val_f1
            best_ep_acc = val_acc

            torch.save(nnmodel.state_dict(), f'best_model_weights/{config.loss_fxn}_{config.model_type}_{config.cutoff}_{config.seed_idx}_{wandb.run.id}.pt')

            # Confusion Matrix

            # Convert val_preds and val_labels to numpy arrays directly
            all_preds = np.array(val_preds)
            all_labels = np.array(val_labels)

            # Threshold the predictions
            binary_preds = (all_preds > 0.5).astype(int)
            binary_labels = all_labels.astype(int)

            cm = confusion_matrix(binary_labels, binary_preds)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            wandb.log({f"{taskname} Confusion Matrix": wandb.Image(plt)})
            plt.close()

        else:
            stop_crit+=1
        if stop_crit>early_stop:
            break

wandb.log({ "total_best_ep": best_ep,
            "total_best_ep_tloss": best_ep_tloss,
            "total_best_ep_vloss": best_ep_vloss,
            "total_best_ep_AUC": best_ep_auc,
            "total_best_ep_precision": best_ep_prec, 
            "total_best_ep_recall": best_ep_recall, 
            "total_best_ep_f1": best_ep_f1,
            "total_best_ep_acc": best_ep_acc,
})

