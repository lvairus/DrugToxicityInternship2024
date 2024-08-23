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
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from binary_nnmodel import NNModel
from data_utils import CustomDataset, RoundRobinBatchSampler
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
# def calc_auc(grnd_truth, predictions):
#     auc_scores = []
#     grnd_truth = np.array(grnd_truth)
#     predictions = np.array(predictions)
#     for i in range(grnd_truth.shape[1]):
#         auc_score = roc_auc_score(grnd_truth[:, i], predictions[:, i])
#         auc_scores.append(auc_score)
    
#     auc_scores_df = pd.DataFrame(auc_scores, columns=["AUC Score"])
#     auc_scores_df.to_csv("auc_scores.csv", index=False) # print out to file
#     # Average AUC scores
#     auc_macro = np.mean(auc_scores)

#     return auc_macro
    

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
            notes = config_dict["init_notes"],
            config = config_dict,
)
config = wandb.config


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


# initialize models
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
LLModel.to("cuda")
LLModel.eval()

nnmodel = NNModel(config).to("cuda")

wandb.watch(nnmodel, log_freq=100)

# ========================================================================================================================

# Data Preprocessing

# import data
data = pd.read_csv(config.dataset)

# split data
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    data[config.smilescol],
    data[config.labelcol],
    test_size=config.testprop,
    shuffle=True,
    stratify=data[config.labelcol],
    random_state=SEED
)

# convert feature pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()
# convert label pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist(), dtype=torch.float)
Y_test = torch.tensor(Y_test.tolist(), dtype=torch.float)

# make one hot label matrices
# Y_hot_train = torch.zeros(Y_train.size(0), Y_train.max() + 1)
# Y_hot_train.scatter_(1, Y_train.unsqueeze(1), 1)
# Y_hot_test = torch.zeros(Y_test.size(0), Y_train.max() + 1)
# Y_hot_test.scatter_(1, Y_test.unsqueeze(1), 1)

# create CustomDataset object
training_dataset = CustomDataset(tokenizer, X_train, Y_train, max_input_length=512, max_target_length=512) # hot?
test_dataset = CustomDataset(tokenizer, X_test, Y_test, max_input_length=512, max_target_length=512) # hot?

# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)

taskname = Path(config.dataset).stem.upper()

# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config['lr'])
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

early_stop = 20
stop_crit = 0
best_ep_auc = 0
best_ep_prec = 0
best_ep_recall = 0
best_ep_f1 = 0
loss_fn = nn.BCELoss()


for epoch in tqdm(range(config.epochs)):
    wandb.log({'epoch': epoch})
    # training
    if True:
        train_running_loss = 0
        # print(f"TYPE LOSSES INIT: {type(train_running_losses)}")
        # print(f"TYPE LOSSES[0]: {type(train_running_losses[0])}")

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

            train_batch_loss = loss
            train_running_loss += loss

            # print(f"TYPE LOSSES: {type(train_batch_losses)}")
            # print(f"TYPE LOSSES[0]: {type(train_batch_losses[0])}")
            # print(f"TYPE TOTAL: {type(train_batch_total_loss)}")
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()

            # wandb.log({'train batch total loss': train_batch_total_loss})
        
        num_train_batches = len(train_dataloader)
        train_avg_loss = train_running_loss / num_train_batches
        wandb.log({'train loss': train_avg_loss})

    # validation
    if True:
        # val_dataloaders = [task[2] for task in tasks]
        # val_dataloaders.remove("skip")
        val_preds = []
        val_labels = []
        val_running_loss = 0
        val_avg_loss = 0

        num_val_minibatches = len(test_dataloader)
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

        val_avg_loss = val_running_loss / num_val_minibatches
        auc = roc_auc_score(val_labels, val_preds) # calc_auc

        binary_preds = (np.array(val_preds) > 0.5).astype(int)
        binary_labels = np.array(val_labels).astype(int)

        val_prec = precision_score(binary_labels, binary_preds)
        val_recall = recall_score(binary_labels, binary_preds)
        val_f1 = f1_score(binary_labels, binary_preds)
        
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

            torch.save(nnmodel.state_dict(), f'model_weights.pt')

            # Confusion Matrix

            # y_true_test = np.argmax(val_labels, axis=1)
            # y_pred_test = np.argmax(val_preds, axis=1)
            # cm = confusion_matrix(y_true_test, y_pred_test)

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
            "total_best_ep_f1": best_ep_f1 
})

# # log bar charts of performance for tasks and total model
# performance_table = wandb.Table(dataframe=performance_df)
# task_performance_table = wandb.Table(dataframe=performance_df.drop("total"))

# # wandb.log({"best_avg_ep" : wandb.plot.bar(performance_table, "task", "best_avg_ep", title="best_avg_ep")}) # unnecessary, already in total_best_ep above, same for all tasks
# wandb.log({"best_avg_ep_tloss" : wandb.plot.bar(task_performance_table, "task", "best_avg_ep_tloss", title="best_avg_ep_tloss")})
# wandb.log({"best_avg_ep_vloss" : wandb.plot.bar(task_performance_table, "task", "best_avg_ep_vloss", title="best_avg_ep_vloss")})
# wandb.log({"best_avg_ep_AUC" : wandb.plot.bar(performance_table, "task", "best_avg_ep_AUC", title="best_avg_ep_AUC")})
# wandb.log({"best_own_ep" : wandb.plot.bar(task_performance_table, "task", "best_own_ep", title="best_own_ep")})
# wandb.log({"best_own_ep_tloss" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_tloss", title="best_own_ep_tloss")})
# wandb.log({"best_own_ep_vloss" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_vloss", title="best_own_ep_vloss")})
# wandb.log({"best_own_ep_AUC" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_AUC", title="best_own_ep_AUC")})

# # columns: "best_avg_ep", "best_avg_ep_tloss", "best_avg_ep_vloss", "best_avg_ep_AUC", "best_own_ep", "best_own_ep_tloss", "best_own_ep_vloss", "best_own_ep_AUC"
# performance_df.to_csv(f"performance_mouse{config.seed_idx}.csv", index=False)