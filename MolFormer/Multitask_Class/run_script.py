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
from classification_layer import NNModel
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

# tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # diff tokenizer
# LLModel = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # diff model

tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
LLModel = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
LLModel.to("cuda")
# for name, layer in LLModel.named_children():
#     print(name, layer)

nnmodel = NNModel(config={"input_size": 768, "embedding_size": 256, "hidden_size": 128, "output_size": 2, "n_layers": 3}).to("cuda")
wandb.watch(nnmodel, log_freq=100)

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
training_set = CustomDataset(tokenizer, X_train, Y_hot_train, max_input_length=512, max_target_length=512)
test_set = CustomDataset(tokenizer, X_test, Y_hot_test, max_input_length=512, max_target_length=512)
# create Dataloader
train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=1e-4)

# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Training Loop
# Output of the model is a dictionary with keys "loss" and "logits"
def train_one_epoch(epoch_index, criterion):
    LLModel.eval()
    running_loss = 0.0
    total_loss = 0
    num_of_examples: int = 0
    # molform_embs_train = []
    # molform_labels_train = []
    for batch in train_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels, 
            #print(outputs)
            #loss = outputs["loss"]
            #logits = outputs["logits"]
            encoder = outputs["hidden_states"][-1]

        # train regression head on top of encoder output to label

        # average over second dimension of encoder output to get a single vector for each example
        encoder = encoder.mean(dim=1)

        # molform_embs_train.append(encoder.cpu().numpy())
        # molform_labels_train.append(y_regression_values.cpu().numpy())

        # pass encoder output to regression head
        nn_outputs = nnmodel(encoder)
        # print(nn_outputs)
        # calculate loss from outputs and ground_truth_y_values
        #nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
        nn_loss = criterion(nn_outputs, y_regression_values)
        total_loss += nn_loss.item()
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        nn_loss.backward()
        optimizer.step()
        running_loss += nn_loss.item()
        if num_of_examples % 10 == 0:
            last_loss = running_loss / 10 # loss per X examples
            # print('num_of_examples {} loss: {} %_data_trained : {}'.format(num_of_examples + 1, last_loss, num_of_examples / len(X_train) * 10))
            # wandb.log({"num_of_examples": num_of_examples, "train_loss": last_loss})
            wandb.log({"loss_lv": nn_loss})
            running_loss = 0.
        num_of_examples += len(batch["input_ids"])
    
    # molform_embs_train = np.concatenate(molform_embs_train, axis=0)
    # molform_labels_train = np.concatenate(molform_labels_train, axis=0)
    # np.save('molform_embs_train.npy', molform_embs_train)
    # np.save('molform_labels_train.npy', molform_labels_train)

    return running_loss

def inference_test_set(epoch_index, criterion):
    LLModel.eval()
    running_tloss = 0.0
    total_tloss = 0
    num_of_examples: int = 0
    # dictionary of all ground_truth and predictions
    outputs_dict = {"ground_truth": [], "predictions": []}
    # molform_embs_test = []
    # molform_labels_test = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        y_regression_values = batch["y_regression_values"]

        with torch.no_grad():
            outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)#labels=labels)
            #print(outputs)
            #loss = outputs["loss"]
            #logits = outputs["logits"]
            encoder = outputs["hidden_states"][-1]

            # inference regression head on top of encoder output to label
            # average over second dimension of encoder output to get a single vector for each example
            encoder = encoder.mean(dim=1)

            # molform_embs_test.append(encoder.cpu().numpy())
            # molform_labels_test.append(y_regression_values.cpu().numpy())

            # pass encoder output to regression head
            nn_outputs = nnmodel(encoder)
            #nn_loss = F.mse_loss(nn_outputs.flatten(), y_regression_values)
            nn_loss = criterion(nn_outputs, y_regression_values)
            # add to dictionary
            outputs_dict["ground_truth"].extend(y_regression_values.cpu().numpy())
            outputs_dict["predictions"].extend(nn_outputs.cpu().detach().numpy())

            total_tloss += nn_loss.item()
            #nn_loss.backward()
            #optimizer.step()
            running_tloss += nn_loss.item()
            if num_of_examples % 10 == 0:
                last_tloss = running_tloss / 100 # loss per X examples
                # print('  num_of_examples {} test_loss: {}'.format(num_of_examples + 1, last_tloss))
                # wandb.log({"num_of_test_examples": num_of_examples, "test_loss": last_tloss})
                wandb.log({"test_loss_lv": nn_loss})
                running_tloss = 0.
                # Track best performance, and save the model's state
                num_of_examples += len(batch["input_ids"])
    
    # molform_embs_test = np.concatenate(molform_embs_test, axis=0)
    # molform_labels_test = np.concatenate(molform_labels_test, axis=0)
    # np.save('molform_embs_test.npy', molform_embs_test)
    # np.save('molform_labels_test.npy', molform_labels_test)

    return outputs_dict

# def generate_parity_plot(ground_truth, predictions):
#     plt.scatter(ground_truth, predictions, s=0.2)
#     # draw line of best fit
#     m, b = np.polyfit(ground_truth, predictions, 1)
#     plt.plot(ground_truth, [m* g + b for g in ground_truth])#m*ground_truth + b)
#     # add labels of correlation coefficient
#     # correlation coefficient
#     r = np.corrcoef(ground_truth, predictions)[0, 1]
#     # pearson's r squared
#     r2 = sklearn.metrics.r2_score(ground_truth, predictions)
#     plt.legend(["Data", "y = {:.2f}x + {:.2f}; r={}; r2={}".format(m, b, r, r2)], loc="upper left")
#     plt.xlabel("Ground Truth")
#     plt.ylabel("Predictions")
#     plt.title("Ground Truth vs Predictions")
#     plt.savefig("parity_plot.png")


# Training loop
epoch_number = 0
best_auc = 0
early_stop = 20
stop_crit = 0
criterion = nn.CrossEntropyLoss()
training_df = {"train_loss":[], "val-auc": []}

old_stdout = sys.stdout
log_file = open("logfile.log","w")

for epoch in (range(args.epochs)):
    sys.stdout = log_file
    trainloss = train_one_epoch(epoch, criterion)
    training_df['train_loss'].append(trainloss)
    outputs_dict = inference_test_set(epoch, criterion)
    epoch_number += 1
    ep_auc = calc_auc(outputs_dict["ground_truth"], outputs_dict["predictions"]) #sklearn.metrics.r2_score(outputs_dict["ground_truth"], outputs_dict["predictions"])
    training_df['val-auc'].append(ep_auc)
    wandb.log({"epoch": epoch, "val-auc": ep_auc})

    print(f'Test auc: {ep_auc}')

    if ep_auc>best_auc:
        best_auc = ep_auc
        #best_vloss = last_tloss
        #model_path = 'model_{}'.format(timestamp)
        #model_scripted = torch.jit.script(nnmodel)
        #model_scripted.save(f'model_{timestamp}.pt')
        torch.save(nnmodel.state_dict(), f'model_weights.pt')
        #del(model_scripted)
        #if epoch>0.75*args.epochs:
        #    # Generate Parity Plot
        #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
        y_true_test = np.argmax(outputs_dict["ground_truth"], axis=1)
        y_pred_test = np.argmax(outputs_dict["predictions"], axis=1)
        cm = confusion_matrix(y_true_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.show()
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        print("Confusion Matrix:\n", cm)
    else:
       stop_crit+=1
    if stop_crit>early_stop:
        break

    sys.stdout = old_stdout

log_file.close()

training_pddf = pd.DataFrame(training_df)
training_pddf.to_csv('training_df.csv')
# Generate Parity Plot
#generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])
