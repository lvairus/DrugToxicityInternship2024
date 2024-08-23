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
from multiclass_nnmodel import NNModel
from data_utils_epacatidx import CustomDataset, RoundRobinBatchSampler
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
    "-y", "--yaml", type=Path, required=False, default="multiclass_config.yaml", help="path to config .yaml file"
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
# cutoff_to_category = {10:0, 50:1, 500:2, 2000:3}
# cutoff_idx = cutoff_to_category[config.cutoff]

# define train and test sets
X_train = train_df['SMILES']
X_test = test_df["SMILES"]
Y_train = train_df['EPACategoryIndex']
Y_test = test_df["EPACategoryIndex"]

# get counts of each category
num_0 = len(Y_train[Y_train == 0])
num_1 = len(Y_train[Y_train == 1])
num_2 = len(Y_train[Y_train == 2])
num_3 = len(Y_train[Y_train == 3])
num_4 = len(Y_train[Y_train == 4])
num_total = len(Y_train)

# convert feature pandas dataframe to list for tokenization
X_train = X_train.tolist()
X_test = X_test.tolist()
# convert label pandas dataframe to tensor
Y_train = torch.tensor(Y_train.tolist())
Y_test = torch.tensor(Y_test.tolist())

# make one hot label matrices
Y_hot_train = torch.zeros(Y_train.size(0), Y_train.max() + 1)
Y_hot_train.scatter_(1, Y_train.unsqueeze(1), 1)
Y_hot_test = torch.zeros(Y_test.size(0), Y_train.max() + 1)
Y_hot_test.scatter_(1, Y_test.unsqueeze(1), 1)

# create CustomDataset object
training_dataset = CustomDataset(tokenizer, X_train, Y_hot_train, Y_train, max_input_length=512, max_target_length=512)
test_dataset = CustomDataset(tokenizer, X_test, Y_hot_test, Y_train, max_input_length=512, max_target_length=512)

# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, worker_init_fn=seed_worker, generator=gen, shuffle=False)


# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config['lr'])
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


# initialize helper variables
classes = ["0-10", "10-50", "50-500", "500-2000", "2000+"]
# prepare performance dataframe
performance_df = pd.DataFrame(index=classes + ["average"],
                              columns=["class",
                                       "best_avg_ep", "best_avg_ep_AUC", "best_avg_ep_prec", "best_avg_ep_recall", "best_avg_ep_f1", 
                                       "best_own_ep", "best_own_ep_AUC", "best_own_ep_prec", "best_own_ep_recall", "best_own_ep_f1", ])
performance_df = performance_df.fillna(0.0)
performance_df = performance_df.infer_objects() 
performance_df["best_avg_ep"] = performance_df["best_avg_ep"].astype(int)
performance_df["best_own_ep"] = performance_df["best_own_ep"].astype(int)
performance_df["class"] = classes + ["average"]


early_stop = 20
stop_crit = 0
if config.loss_fxn == "weighted":
    class_counts = torch.tensor([num_0, num_1, num_2, num_3, num_4], dtype=torch.float, device="cuda")
    class_weights = num_total / class_counts
    class_weights = class_weights / class_weights.sum()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_fn = nn.CrossEntropyLoss()


for epoch in tqdm(range(config.epochs)):
    wandb.log({'epoch': epoch})
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
        preds = nnmodel(encoder) 
        loss = loss_fn(preds, y_regression_values)

        train_running_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_avg_loss = loss / len(train_dataloader)
    wandb.log({'train loss': train_avg_loss})

    # validation
    val_preds = []
    val_labels = []
    val_running_loss = 0
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        y_regression_values = batch["y_regression_values"]
        with torch.no_grad():
            # molformer
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            encoder = outputs["hidden_states"][-1]
            encoder = encoder.mean(dim=1)
            # nnmodel
            preds  = nnmodel(encoder)
            loss = loss_fn(preds, y_regression_values)

            val_running_loss += loss
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_regression_values.cpu().numpy())

    val_avg_loss = val_running_loss / len(test_dataloader)
    wandb.log({'val loss': val_avg_loss})

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    auc_scores = [0] * len(classes)
    for i in range(val_labels.shape[1]):
        auc_score = roc_auc_score(val_labels[:, i], val_preds[:, i])
        auc_scores[i] = auc_score
    binary_preds = (val_preds > 0.5).astype(int)
    binary_labels = val_labels.astype(int)
    prec_scores = precision_score(binary_labels, binary_preds, average=None, zero_division=0)
    recall_scores = recall_score(binary_labels, binary_preds, average=None)
    f1_scores = f1_score(binary_labels, binary_preds, average=None)

    auc_avg = np.mean(auc_scores)
    prec_avg = np.mean(prec_scores)
    recall_avg = np.mean(recall_scores)
    f1_avg = np.mean(f1_scores)

    wandb.log({"val auc": auc_avg,
               "val prec": prec_avg,
               "val recall": recall_avg,
               "val f1": f1_avg,
               })
    for i, (auc, prec, recall, f1) in enumerate(zip(auc_scores, prec_scores, recall_scores, f1_scores)):
        wandb.log({f"class {i} auc": auc,
                   f"class {i} prec": prec,
                   f"class {i} recall": recall,
                   f"class {i} f1": f1,
                   })

    # update performance_df if any f1s got better
    for i, taskname in enumerate(f1_scores):
        if f1_scores[i] > performance_df.iloc[i]["best_own_ep_f1"]:
            performance_df.loc[classes[i], "best_own_ep"] = epoch
            performance_df.loc[classes[i], "best_own_ep_AUC"] = float(auc_scores[i])
            performance_df.loc[classes[i], "best_own_ep_prec"] = float(prec_scores[i])
            performance_df.loc[classes[i], "best_own_ep_recall"] = float(recall_scores[i])
            performance_df.loc[classes[i], "best_own_ep_f1"] = float(f1_scores[i])

    # take out specific tasks if their auc decreases early_stop times
    # save weights of specific last layers if their auc increases

    # early stopping and saving best results
    if f1_avg>performance_df.loc["average", "best_avg_ep_f1"]:
        stop_crit = 0
        performance_df.loc["average", "best_avg_ep"] = epoch
        performance_df.loc["average", "best_avg_ep_AUC"] = auc_avg
        performance_df.loc["average", "best_avg_ep_prec"] = prec_avg
        performance_df.loc["average", "best_avg_ep_recall"] = recall_avg
        performance_df.loc["average", "best_avg_ep_f1"] = f1_avg

        for i in range(len(classes)):
            performance_df.loc[classes[i], "best_avg_ep"] = epoch
            performance_df.loc[classes[i], "best_avg_ep_AUC"] = auc_scores[i]
            performance_df.loc[classes[i], "best_avg_ep_prec"] = prec_scores[i]
            performance_df.loc[classes[i], "best_avg_ep_recall"] = recall_scores[i]
            performance_df.loc[classes[i], "best_avg_ep_f1"] = f1_scores[i]

        torch.save(nnmodel.state_dict(), f'best_model_weights/{config.loss_fxn}_{config.model_type}_{config.seed_idx}_{wandb.run.id}.pt')

        # confusion matrix

        # y_true_test = np.argmax(val_labels, axis=1)
        # y_pred_test = np.argmax(val_preds, axis=1)
        # cm = confusion_matrix(y_true_test, y_pred_test)

        y_true = np.argmax(val_labels, axis=1)
        y_pred = np.argmax(val_preds, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{config.model_type} Confusion Matrix')
        wandb.log({f"{config.model_type} Confusion Matrix": wandb.Image(plt)})
        plt.close()

    else:
        stop_crit+=1
    if stop_crit>early_stop:
        break

wandb.log({ "total_best_ep": performance_df.loc["average", "best_avg_ep"],
            "total_best_ep_AUC": performance_df.loc["average", "best_avg_ep_AUC"],
            "total_best_ep_prec": performance_df.loc["average", "best_avg_ep_prec"],
            "total_best_ep_recall": performance_df.loc["average", "best_avg_ep_recall"],
            "total_best_ep_f1": performance_df.loc["average", "best_avg_ep_f1"],
})


# log bar charts of performance for tasks and total model
performance_table = wandb.Table(dataframe=performance_df)
task_performance_table = wandb.Table(dataframe=performance_df.drop("average"))

# wandb.log({"best_avg_ep" : wandb.plot.bar(performance_table, "task", "best_avg_ep", title="best_avg_ep")}) # unnecessary, already in total_best_ep above, same for all tasks
wandb.log({"best_avg_ep_AUC" : wandb.plot.bar(performance_table, "class", "best_avg_ep_AUC", title="best_avg_ep_AUC")})
wandb.log({"best_avg_ep_prec" : wandb.plot.bar(performance_table, "class", "best_avg_ep_prec", title="best_avg_ep_prec")})
wandb.log({"best_avg_ep_recall" : wandb.plot.bar(performance_table, "class", "best_avg_ep_recall", title="best_avg_ep_recall")})
wandb.log({"best_avg_ep_f1" : wandb.plot.bar(performance_table, "class", "best_avg_ep_f1", title="best_avg_ep_f1")})
wandb.log({"best_own_ep" : wandb.plot.bar(task_performance_table, "class", "best_own_ep", title="best_own_ep")})
wandb.log({"best_own_ep_AUC" : wandb.plot.bar(task_performance_table, "class", "best_own_ep_AUC", title="best_own_ep_AUC")})
wandb.log({"best_own_ep_prec" : wandb.plot.bar(task_performance_table, "class", "best_own_ep_prec", title="best_own_ep_prec")})
wandb.log({"best_own_ep_recall" : wandb.plot.bar(task_performance_table, "class", "best_own_ep_recall", title="best_own_ep_recall")})
wandb.log({"best_own_ep_f1" : wandb.plot.bar(task_performance_table, "class", "best_own_ep_f1", title="best_own_ep_f1")})

performance_df.to_csv(f"performance_df_check.csv", index=False)

# performance_df = pd.DataFrame(index=classes + ["average"], 
#                               columns=["class",
#                                        "best_avg_ep", "best_avg_ep_AUC", "best_avg_ep_prec", "best_avg_ep_recall", "best_avg_ep_f1", 
#                                        "best_own_ep", "best_own_ep_AUC", "best_own_ep_prec", "best_own_ep_recall", "best_own_ep_f1", ])