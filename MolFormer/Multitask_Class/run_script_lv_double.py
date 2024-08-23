# imports
# Load model directly
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from classification_layer_multi import NNModel
from data_utils import CustomDataset
import sys
import pdb
import wandb
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, SUPPRESS
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMaskedLM


# Calculate and avg AUC for each class
def calc_auc(grnd_truth, predictions):
    auc_scores = []
    grnd_truth = np.array(grnd_truth)
    predictions = np.array(predictions)
    for i in range(grnd_truth.shape[1]):
        auc_score = roc_auc_score(grnd_truth[:, i], predictions[:, i])
        auc_scores.append(auc_score)
    
    auc_scores_df = pd.DataFrame(auc_scores, columns=["AUC Score"])
    auc_scores_df.to_csv("auc_scores.csv", index=False)
    # Average AUC scores
    auc_macro = np.mean(auc_scores)

    return auc_macro
    
# parse arguments
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-d", "--dataset", type=Path, required=True, help="Input data for training/validation"
)

parser.add_argument(
    "-s", "--smilescol", type=str, required=True, help="Column for SMILES"
)

parser.add_argument(
    "-l", "--labelcol", type=str, required=True, help="Column for labels"
)

parser.add_argument(
    "-t", "--testprop", type=float, required=True, help="Proportion of data used for training"
)

parser.add_argument(
    "-E", "--epochs", type=int, required=True, help="Number of epochs"
)

args = parser.parse_args()


# init wandb to log results
wandb.init()

tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
LLModel.to("cuda")
# for name, layer in LLModel.named_children():
#     print(name, layer)

nnmodel = NNModel(config={"input_size": 768, "embedding_size": 256, "hidden_size": 128, "output_size": 5, "n_layers": 3}).to("cuda")
wandb.watch(nnmodel, log_freq=100)


# Data Preprocessing
# import data
# split data
# convert to lists and tensors
# make one hots
# make Dataset
# make Dataloader

# import data
data_rat = pd.read_csv('oral_data/oral_rat.csv')
data_mouse = pd.read_csv('oral_data/oral_mouse.csv')

# split data
X_train_rat, X_test_rat = sklearn.model_selection.train_test_split(data_rat[args.smilescol], test_size=args.testprop, random_state=42) # features
Y_train_rat, Y_test_rat = sklearn.model_selection.train_test_split(data_rat[args.labelcol], test_size=args.testprop, random_state=42) # labels

X_train_mouse, X_test_mouse = sklearn.model_selection.train_test_split(data_mouse[args.smilescol], test_size=args.testprop, random_state=42) # features
Y_train_mouse, Y_test_mouse = sklearn.model_selection.train_test_split(data_mouse[args.labelcol], test_size=args.testprop, random_state=42) # labels

# convert feature pandas dataframe to list for tokenization
X_train_rat = X_train_rat.tolist()
X_test_rat = X_test_rat.tolist()
X_train_mouse = X_train_mouse.tolist()
X_test_mouse = X_test_mouse.tolist()
# convert label pandas dataframe to tensor
Y_train_rat = torch.tensor(Y_train_rat.tolist())
Y_test_rat = torch.tensor(Y_test_rat.tolist())
Y_train_mouse = torch.tensor(Y_train_mouse.tolist())
Y_test_mouse = torch.tensor(Y_test_mouse.tolist())

# make one hot label matrices
Y_hot_train_rat = torch.zeros(Y_train_rat.size(0), Y_train_rat.max() + 1)
Y_hot_train_rat.scatter_(1, Y_train_rat.unsqueeze(1), 1)
Y_hot_test_rat = torch.zeros(Y_test_rat.size(0), Y_train_rat.max() + 1)
Y_hot_test_rat.scatter_(1, Y_test_rat.unsqueeze(1), 1)
Y_hot_train_mouse = torch.zeros(Y_train_mouse.size(0), Y_train_mouse.max() + 1)
Y_hot_train_mouse.scatter_(1, Y_train_mouse.unsqueeze(1), 1)
Y_hot_test_mouse = torch.zeros(Y_test_mouse.size(0), Y_train_mouse.max() + 1)
Y_hot_test_mouse.scatter_(1, Y_test_mouse.unsqueeze(1), 1)

# create CustomDataset object
training_dataset_rat = CustomDataset(tokenizer, X_train_rat, Y_hot_train_rat, max_input_length=512, max_target_length=512)
test_dataset_rat = CustomDataset(tokenizer, X_test_rat, Y_hot_test_rat, max_input_length=512, max_target_length=512)
training_dataset_mouse = CustomDataset(tokenizer, X_train_mouse, Y_hot_train_mouse, max_input_length=512, max_target_length=512)
test_dataset_mouse = CustomDataset(tokenizer, X_test_mouse, Y_hot_test_mouse, max_input_length=512, max_target_length=512)
# create Dataloader
train_dataloader_rat = torch.utils.data.DataLoader(training_dataset_rat, batch_size=128, shuffle=True)
test_dataloader_rat = torch.utils.data.DataLoader(test_dataset_rat, batch_size=128, shuffle=False)
train_dataloader_mouse = torch.utils.data.DataLoader(training_dataset_mouse, batch_size=128, shuffle=True)
test_dataloader_mouse = torch.utils.data.DataLoader(test_dataset_mouse, batch_size=128, shuffle=False)


# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=1e-4)
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# initialize helper variables
early_stop = 20
stop_crit = 0
best_auc = 0
loss_fn = nn.CrossEntropyLoss()

for i in tqdm(range(args.epochs)):
    LLModel.eval()
    # training
    for batch_rat, batch_mouse in zip(train_dataloader_rat, train_dataloader_mouse):
        # rat loss
        # pass through Molformer
        input_ids_rat = batch_rat["input_ids"]
        attention_mask_rat = batch_rat["attention_mask"]
        labels_rat = batch_rat["labels"]
        y_regression_values_rat = batch_rat["y_regression_values"]
        with torch.no_grad():
            outputs_rat = LLModel(input_ids=input_ids_rat, attention_mask=attention_mask_rat, output_hidden_states=True)#labels=labels, 
            encoder_rat = outputs_rat["hidden_states"][-1]
        # average over second dimension of encoder output to get a single vector for each example
        encoder_rat = encoder_rat.mean(dim=1)
        # pass through our model
        preds_rat = nnmodel(encoder_rat, 10) 
        loss_rat = loss_fn(preds_rat, y_regression_values_rat)

        # mouse loss
        # pass through Molformer
        input_ids_mouse = batch_mouse["input_ids"]
        attention_mask_mouse = batch_mouse["attention_mask"]
        labels_mouse = batch_mouse["labels"]
        y_regression_values_mouse = batch_mouse["y_regression_values"]
        with torch.no_grad():
            outputs_mouse = LLModel(input_ids=input_ids_mouse, attention_mask=attention_mask_mouse, output_hidden_states=True)#labels=labels, 
            encoder_mouse = outputs_mouse["hidden_states"][-1]
        # average over second dimension of encoder output to get a single vector for each example
        encoder_mouse = encoder_mouse.mean(dim=1)
        # pass through our model
        preds_mouse = nnmodel(encoder_mouse, 10) 
        loss_mouse = loss_fn(preds_mouse, y_regression_values_mouse)

        # (multitask) sum loss with other tasks
        loss_total = loss_rat + loss_mouse

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    wandb.log({'train rat loss': loss_rat})
    wandb.log({'train mouse loss': loss_mouse})
    wandb.log({'train total loss': loss_total})

    # validation
    val_preds_rat = []
    val_labels_rat = []
    val_preds_mouse = []
    val_labels_mouse = []
    for batch_rat, batch_mouse in zip(test_dataloader_rat, test_dataloader_mouse):
        input_ids_rat = batch_rat["input_ids"]
        attention_mask_rat = batch_rat["attention_mask"]
        labels_rat = batch_rat["labels"]
        y_regression_values_rat = batch_rat["y_regression_values"]
        with torch.no_grad():
            outputs_rat = LLModel(input_ids=input_ids_rat, attention_mask=attention_mask_rat, output_hidden_states=True)#labels=labels, 
            encoder_rat = outputs_rat["hidden_states"][-1]
            encoder_rat = encoder_rat.mean(dim=1)

            preds_rat  = nnmodel(encoder_rat, 10)
            loss_rat = loss_fn(preds_rat, y_regression_values_rat)

            val_preds_rat.extend(preds_rat.cpu().numpy())
            val_labels_rat.extend(y_regression_values_rat.cpu().numpy())

        input_ids_mouse = batch_mouse["input_ids"]
        attention_mask_mouse = batch_mouse["attention_mask"]
        labels_mouse = batch_mouse["labels"]
        y_regression_values_mouse = batch_mouse["y_regression_values"]
        with torch.no_grad():
            outputs_mouse = LLModel(input_ids=input_ids_mouse, attention_mask=attention_mask_mouse, output_hidden_states=True)#labels=labels, 
            encoder_mouse = outputs_mouse["hidden_states"][-1]
            encoder_mouse = encoder_mouse.mean(dim=1)

            preds_mouse  = nnmodel(encoder_mouse, 7)
            loss_mouse = loss_fn(preds_mouse, y_regression_values_mouse)

            val_preds_mouse.extend(preds_mouse.cpu().numpy())
            val_labels_mouse.extend(y_regression_values_mouse.cpu().numpy())

    loss_total = loss_rat + loss_mouse
    wandb.log({'val rat loss': loss_rat})
    wandb.log({'val mouse loss': loss_mouse})
    wandb.log({'val total loss': loss_total})

    val_auc_rat = calc_auc(val_labels_rat, val_preds_rat)
    val_auc_mouse = calc_auc(val_labels_mouse, val_preds_mouse)
    val_auc_avg = (val_auc_rat + val_auc_mouse) / 2
    wandb.log({'val rat auc': val_auc_rat})
    wandb.log({'val mouse auc': val_auc_mouse})
    wandb.log({'val avg auc': val_auc_avg})

    # early stopping and saving best results
    if val_auc_avg > best_auc:
        # stop_crit = 0 ?
        best_auc = val_auc_avg
        #best_vloss = last_tloss
        #model_path = 'model_{}'.format(timestamp)
        #model_scripted = torch.jit.script(nnmodel)
        #model_scripted.save(f'model_{timestamp}.pt')
        torch.save(nnmodel.state_dict(), f'model_weights.pt')
        #del(model_scripted)
        #if epoch>0.75*args.epochs:
        #    # Generate Parity Plot
        #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
        y_true_test_rat = np.argmax(val_labels_rat, axis=1)
        y_pred_test_rat = np.argmax(val_preds_rat, axis=1)
        cm_rat = confusion_matrix(y_true_test_rat, y_pred_test_rat)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_rat, annot=True, fmt='d', cmap='Blues')
        wandb.log({"Rat Confusion Matrix": wandb.Image(plt)})

        y_true_test_mouse = np.argmax(val_labels_mouse, axis=1)
        y_pred_test_mouse = np.argmax(val_preds_mouse, axis=1)
        cm_mouse = confusion_matrix(y_true_test_mouse, y_pred_test_mouse)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_mouse, annot=True, fmt='d', cmap='Blues')
        wandb.log({"Mouse Confusion Matrix": wandb.Image(plt)})
    else:
       stop_crit+=1
    if stop_crit>early_stop:
        break

