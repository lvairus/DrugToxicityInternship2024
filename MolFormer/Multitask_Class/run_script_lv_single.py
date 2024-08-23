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
from classification_layer_single import NNModel
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

nnmodel = NNModel(config={"input_size": 768, "embedding_size": 256, "hidden_size": 256, "output_size": 5, "n_layers": 3}).to("cuda")
wandb.watch(nnmodel, log_freq=100)


# ===================================================================================================

# Data Preprocessing

# import data
data = pd.read_csv(args.dataset)

# split data
X_train, X_test = sklearn.model_selection.train_test_split(data[args.smilescol], test_size=args.testprop, random_state=42) # features
Y_train, Y_test = sklearn.model_selection.train_test_split(data[args.labelcol], test_size=args.testprop, random_state=42) # labels

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
training_dataset = CustomDataset(tokenizer, X_train, Y_hot_train, max_input_length=512, max_target_length=512)
test_dataset = CustomDataset(tokenizer, X_test, Y_hot_test, max_input_length=512, max_target_length=512)

# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=256, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

# ===================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=1e-4)
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# initialize helper variables
early_stop = 20
stop_crit = 0
best_auc = 0
loss_fn = nn.CrossEntropyLoss()

# Epoch loop
for epoch in tqdm(range(args.epochs)):
    LLModel.eval()
    # training
    for batch in train_dataloader:
        # pass through Molformer
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]
        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels, 
            encoder = outputs["hidden_states"][-1]
        # average over second dimension of encoder output to get a single vector for each example
        encoder = encoder.mean(dim=1)
        # pass through our model
        preds = nnmodel(encoder, 10) 
        loss = loss_fn(preds, y_regression_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wandb.log({'BIRD train loss': loss})

    # validation
    val_preds = []
    val_labels = []
    running_loss = 0
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]
        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels, 
            encoder = outputs["hidden_states"][-1]
            encoder = encoder.mean(dim=1)

            preds  = nnmodel(encoder, 10)
            loss = loss_fn(preds, y_regression_values)
            running_loss += loss

            val_preds.extend(preds.cpu().detach().numpy())
            val_labels.extend(y_regression_values.cpu().numpy())

    num_val_batches = len(test_dataloader)
    avg_loss = running_loss / num_val_batches
    wandb.log({'BIRD val loss': avg_loss})

    val_auc = calc_auc(val_labels, val_preds)
    wandb.log({'BIRD val auc': val_auc})

    # early stopping and saving best results
    if val_auc > best_auc:
        # stop_crit = 0 ?
        best_auc = val_auc
        #best_vloss = last_tloss
        #model_path = 'model_{}'.format(timestamp)
        #model_scripted = torch.jit.script(nnmodel)
        #model_scripted.save(f'model_{timestamp}.pt')
        torch.save(nnmodel.state_dict(), f'model_weights.pt')
        #del(model_scripted)
        #if epoch>0.75*args.epochs:
        #    # Generate Parity Plot
        #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
        y_true_test = np.argmax(val_labels, axis=1)
        y_pred_test = np.argmax(val_preds, axis=1)
        cm = confusion_matrix(y_true_test, y_pred_test)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        wandb.log({"BIRD Confusion Matrix": wandb.Image(plt)})
        plt.close()
    else:
       stop_crit+=1
    if stop_crit>early_stop:
        break

