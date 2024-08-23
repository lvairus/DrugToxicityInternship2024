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
import glob
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
    "-y", "--yaml", type=Path, required=False, default="inference_config.yaml", help="path to config .yaml file"
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
model_type = config.model_type
cutoff = config.cutoff
seed_idx = config.seed_idx

# ========================================================================================================================

# Reproducability
seeds = [53844, 837465, 800662, 910250, 543584, 179839, 707873, 482701, 278083, 198125]
SEED = seeds[seed_idx]

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

file_list = glob.glob(f"best_model_weights/{model_type}_{cutoff}_{seed_idx}*.pt")
file_path = file_list[0] 

nnmodel = NNModel(config).to("cuda")
checkpoint = torch.load(file_path)
nnmodel.load_state_dict(checkpoint)
# models[f"{type}_{cutoff}_{seed_idx}"] = nnmodel

wandb.watch(nnmodel, log_freq=100)


# ========================================================================================================================

# Data Preprocessing

# load in pre-split data 
test_df = pd.read_csv("data/mouse_test_df.csv")
cutoff_to_category = {10:0, 50:1, 500:2, 2000:3}
cutoff_idx = cutoff_to_category[cutoff]

# if tree model and not the root, load in specific input data
if model_type == "tree":
    if cutoff != 500:
        test_df = pd.read_csv(f"data/tree_input_{cutoff}_{seed_idx}.csv")

# make binary column 
test_df["tox"] =  (test_df["EPACategoryIndex"] <= cutoff_idx).astype(int)

# define train and test sets
X_test = test_df["SMILES"]
Y_test = test_df["tox"]
epacatidxs_test = test_df["EPACategoryIndex"]

# convert feature pandas dataframe to list for tokenization
X_test = X_test.tolist()
# convert label pandas dataframe to tensor
Y_test = torch.tensor(Y_test.tolist(), dtype=torch.float)
epacatidxs_test = torch.tensor(epacatidxs_test.tolist(), dtype=torch.int)

# create CustomDataset object
test_dataset = CustomDataset(tokenizer=tokenizer, data=X_test, y_regression_values=Y_test, 
                            epacatidxs=epacatidxs_test, max_input_length=512, max_target_length=512) 

# create Dataloader
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)

taskname = config.init_group

# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config['lr'])
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

loss_fn = nn.BCELoss()

# Validation
smiles = []
epacatidxs = []
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
        epacatidxs.extend(batch["epacatidxs"].cpu().numpy())
        
        decoded_smiles = [tokenizer.decode(ids.cpu(), skip_special_tokens=True) for ids in input_ids]
        smiles.extend(decoded_smiles)

        val_running_loss += loss

# print(smiles)

val_avg_loss = val_running_loss / len(test_dataloader)
auc = roc_auc_score(val_labels, val_preds) # calc_auc

binary_preds = (np.array(val_preds) > 0.5).astype(int)
binary_labels = np.array(val_labels).astype(int)
val_prec = precision_score(binary_labels, binary_preds, zero_division=0)
val_recall = recall_score(binary_labels, binary_preds)
val_f1 = f1_score(binary_labels, binary_preds)
val_acc = accuracy_score(binary_labels, binary_preds)

# # saving tree data splits
# # Convert your lists to a DataFrame
# node_500_results = pd.DataFrame({
#     'SMILES': smiles,
#     'preds': binary_preds.flatten(),
#     'labels': binary_labels.flatten(),
#     'EPACategoryIndex': epacatidxs,
# })
# # split into pos and neg results
# node_50_input = node_500_results[node_500_results["preds"]==1]
# node_2000_input = node_500_results[node_500_results["preds"]==0]
# # Save the DataFrame to a CSV file
# node_50_input.to_csv(f'data/tree_input_50_{seed_idx}.csv', index=False)
# node_2000_input.to_csv(f'data/tree_input_2000_{seed_idx}.csv', index=False)


# Confusion Matrix

# Convert val_preds and val_labels to numpy arrays directly
all_preds = np.array(val_preds)
all_labels = np.array(val_labels)
# Threshold the predictions
binary_preds = (all_preds > 0.5).astype(int)
binary_labels = all_labels.astype(int)
# plot confusion matrix
cm = confusion_matrix(binary_labels, binary_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
wandb.log({f"{model_type} Confusion Matrix": wandb.Image(plt)})
plt.close()

# Extract TN, FP, FN, TP from the confusion matrix
TN, FP, FN, TP = cm.ravel()
# Calculate False Positive Rate (FPR)
FPR = FP / (FP + TN)
# Calculate True Negative Rate (TNR)
TNR = TN / (TN + FP)

# log metrics
wandb.log({'val loss': val_avg_loss})
wandb.log({'val auc': auc})
wandb.log({'val precision': val_prec})
wandb.log({'val recall': val_recall})
wandb.log({'val f1': val_f1})
wandb.log({'val acc': val_acc})
wandb.log({"false pos rate": FPR})
wandb.log({"true neg rate": TNR})

