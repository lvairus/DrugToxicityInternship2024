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
from multitask_nnmodel import NNModel
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

# Calculate and avg AUC for each class
def calc_auc(grnd_truth, predictions):
    auc_scores = []
    grnd_truth = np.array(grnd_truth)
    predictions = np.array(predictions)
    for i in range(grnd_truth.shape[1]):
        auc_score = roc_auc_score(grnd_truth[:, i], predictions[:, i])
        auc_scores.append(auc_score)
    
    auc_scores_df = pd.DataFrame(auc_scores, columns=["AUC Score"])
    auc_scores_df.to_csv("auc_scores.csv", index=False) # print out to file
    # Average AUC scores
    auc_macro = np.mean(auc_scores)

    return auc_macro

# ========================================================================================================================

# parse arguments: read in yaml file with all hyperparameters
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-y", "--yaml", type=Path, required=False, default="multitask_config.yaml", help="path to config .yaml file"
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

# make datasets and dataloaders for each task

cutoffs = ["10","50","500","2000"]
cutoff_to_category = {10:0, 50:1, 500:2, 2000:3}
cutoff_idx = cutoff_to_category[config.cutoff]

# filter data based on tree 
train_df_500 = train_df.copy()
test_df_500 = test_df.copy()

train_df_10 = train_df[train_df['EPACategoryIndex'] <= 1]
test_df_10 = test_df[test_df['EPACategoryIndex'] <= 1]

train_df_50 = train_df[train_df['EPACategoryIndex'] <= 2]
test_df_50 = test_df[test_df['EPACategoryIndex'] <= 2]
train_df_2000 = train_df[train_df['EPACategoryIndex'] >= 3]
test_df_2000 = test_df[test_df['EPACategoryIndex'] >= 3]

train_dfs = [train_df_10,train_df_50,train_df_500,train_df_2000]
test_dfs = [test_df_10,test_df_50,test_df_500,test_df_2000]

train_dataloaders = [None] * 4
test_dataloaders = [None] * 4

for i, (train_df, test_df) in enumerate(zip(train_dfs, test_dfs)):
    # define train and test sets
    X_train = train_df["SMILES"]
    X_test = test_df["SMILES"]
    Y_train = train_df[f"tox{cutoffs[i]}"]
    Y_test = test_df[f"tox{cutoffs[i]}"]

    # convert feature pandas dataframe to list for tokenization
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    # convert label pandas dataframe to tensor
    Y_train = torch.tensor(Y_train.tolist(), dtype=torch.float)
    Y_test = torch.tensor(Y_test.tolist(), dtype=torch.float)

    # create CustomDataset object
    training_dataset = CustomDataset(tokenizer, X_train, Y_train, Y_train, max_input_length=512, max_target_length=512) 
    test_dataset = CustomDataset(tokenizer, X_test, Y_test, Y_train, max_input_length=512, max_target_length=512) 

    # create Dataloader
    train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)

    train_dataloaders[i] = train_dataloader
    test_dataloaders[i] = test_dataloader


# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config['lr'])
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

performance_df = pd.DataFrame(index=cutoffs + ["average"],
                              columns=["cutoff",
                                       "best_avg_ep", "best_avg_ep_AUC", "best_avg_ep_prec", "best_avg_ep_recall", "best_avg_ep_f1", 
                                       "best_own_ep", "best_own_ep_AUC", "best_own_ep_prec", "best_own_ep_recall", "best_own_ep_f1", ])
performance_df = performance_df.fillna(0.0)
performance_df = performance_df.infer_objects() 
performance_df["best_avg_ep"] = performance_df["best_avg_ep"].astype(int)
performance_df["best_own_ep"] = performance_df["best_own_ep"].astype(int)
performance_df["cutoff"] = cutoffs + ["average"]

early_stop = 20
stop_crit = 0
best_ep_auc = 0
best_ep_prec = 0
best_ep_recall = 0
best_ep_f1 = 0
loss_fn = nn.BCELoss()


for epoch in tqdm(range(config.epochs)):
    wandb.log({'epoch': epoch})

    train_losses = [0] * 4
    val_losses = [0] * 4
    auc_scores = [0] * 4
    prec_scores = [0] * 4
    recall_scores = [0] * 4
    f1_scores = [0] * 4

    for task_id in range(4):
        train_dataloader = train_dataloaders[task_id]
        test_dataloader = test_dataloaders[task_id]
        cutoff = cutoffs[task_id]
        
        # training
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
            preds = nnmodel(encoder, task_id)            
            loss = loss_fn(preds.flatten(), y_regression_values)
            train_running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_avg_loss = train_running_loss / len(train_dataloader)
        train_losses[task_id] = train_avg_loss
        wandb.log({f'cutoff {cutoff} tloss': train_avg_loss})

        # validation
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

                preds  = nnmodel(encoder, task_id)
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

        val_losses[task_id] = val_avg_loss
        auc_scores[task_id] = auc
        prec_scores[task_id] = val_prec
        recall_scores[task_id] = val_recall
        f1_scores[task_id] = val_f1

        wandb.log({f'cutoff {cutoff} vloss': val_avg_loss, 
                   f'cutoff {cutoff} auc': auc,
                   f'cutoff {cutoff} precision': val_prec,
                   f'cutoff {cutoff} recall': val_recall,
                   f'cutoff {cutoff} f1': val_f1})
        
        # class-wise saving best results
        if val_f1>performance_df.loc[cutoff, "best_own_ep_f1"]:
            performance_df.loc[cutoff, "best_own_ep"] = epoch
            performance_df.loc[cutoff, "best_own_ep_AUC"] = auc_scores[task_id] # float(auc_scores[task_id])
            performance_df.loc[cutoff, "best_own_ep_prec"] = prec_scores[task_id] # float(prec_scores[task_id])
            performance_df.loc[cutoff, "best_own_ep_recall"] = recall_scores[task_id] # float(recall_scores[task_id])
            performance_df.loc[cutoff, "best_own_ep_f1"] = f1_scores[task_id] # float(f1_scores[task_id])

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
            wandb.log({f"{config.model_type} {cutoff} Confusion Matrix": wandb.Image(plt)})
            plt.close()

    avg_train_loss = sum(train_losses) / 4
    avg_val_loss = sum(val_losses) / 4
    avg_auc = sum(auc_scores) / 4
    avg_prec = sum(prec_scores) / 4
    avg_recall = sum(recall_scores) / 4
    avg_f1 = sum(f1_scores) / 4

    wandb.log({'train loss': avg_train_loss, 
               'val loss': avg_val_loss,
               'val auc': avg_auc,
               'val precision': avg_prec,
               'val recall': avg_recall,
               'val f1': avg_f1})

    # early stopping and saving best results
    if avg_f1>best_ep_f1:
        stop_crit = 0
        performance_df.loc["average", "best_avg_ep"] = epoch
        performance_df.loc["average", "best_avg_ep_AUC"] = avg_auc
        performance_df.loc["average", "best_avg_ep_prec"] = avg_prec
        performance_df.loc["average", "best_avg_ep_recall"] = avg_recall
        performance_df.loc["average", "best_avg_ep_f1"] = avg_f1

        for i in range(len(cutoffs)):
            performance_df.loc[cutoffs[i], "best_avg_ep"] = epoch
            performance_df.loc[cutoffs[i], "best_avg_ep_AUC"] = auc_scores[i]
            performance_df.loc[cutoffs[i], "best_avg_ep_prec"] = prec_scores[i]
            performance_df.loc[cutoffs[i], "best_avg_ep_recall"] = recall_scores[i]
            performance_df.loc[cutoffs[i], "best_avg_ep_f1"] = f1_scores[i]
    
        torch.save(nnmodel.state_dict(), f'best_model_weights/{config.init_group}_{config.cutoff}_{config.seed_idx}_{wandb.run.id}.pt')

    else:
        stop_crit+=1
    if stop_crit>early_stop:
        break

# log best avg performance
wandb.log({ "total_best_ep": performance_df.loc["average", "best_avg_ep"],
            "total_best_ep_AUC": performance_df.loc["average", "best_avg_ep_AUC"],
            "total_best_ep_prec": performance_df.loc["average", "best_avg_ep_prec"],
            "total_best_ep_recall": performance_df.loc["average", "best_avg_ep_recall"],
            "total_best_ep_f1": performance_df.loc["average", "best_avg_ep_f1"],
})

# log bar charts of performance for cutoffs and total model
performance_table = wandb.Table(dataframe=performance_df)
task_performance_table = wandb.Table(dataframe=performance_df.drop("average"))

# wandb.log({"best_avg_ep" : wandb.plot.bar(performance_table, "task", "best_avg_ep", title="best_avg_ep")}) # unnecessary, already in total_best_ep above, same for all tasks
wandb.log({"best_avg_ep_AUC" : wandb.plot.bar(performance_table, "cutoff", "best_avg_ep_AUC", title="best_avg_ep_AUC")})
wandb.log({"best_avg_ep_prec" : wandb.plot.bar(performance_table, "cutoff", "best_avg_ep_prec", title="best_avg_ep_prec")})
wandb.log({"best_avg_ep_recall" : wandb.plot.bar(performance_table, "cutoff", "best_avg_ep_recall", title="best_avg_ep_recall")})
wandb.log({"best_avg_ep_f1" : wandb.plot.bar(performance_table, "cutoff", "best_avg_ep_f1", title="best_avg_ep_f1")})
wandb.log({"best_own_ep" : wandb.plot.bar(task_performance_table, "cutoff", "best_own_ep", title="best_own_ep")})
wandb.log({"best_own_ep_AUC" : wandb.plot.bar(task_performance_table, "cutoff", "best_own_ep_AUC", title="best_own_ep_AUC")})
wandb.log({"best_own_ep_prec" : wandb.plot.bar(task_performance_table, "cutoff", "best_own_ep_prec", title="best_own_ep_prec")})
wandb.log({"best_own_ep_recall" : wandb.plot.bar(task_performance_table, "cutoff", "best_own_ep_recall", title="best_own_ep_recall")})
wandb.log({"best_own_ep_f1" : wandb.plot.bar(task_performance_table, "cutoff", "best_own_ep_f1", title="best_own_ep_f1")})

performance_df.to_csv(f"performance_df_check.csv", index=False)

