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
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from classification_layer_multi import NNModel
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
    

# parse arguments: read in yaml file with all hyperparameters
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-y", "--yaml", type=Path, required=False, default="multi.yaml", help="path to config .yaml file"
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
# took out human data bc it gives errors

len_smallest_dataset = 121
len_smallest_testset = math.ceil(len_smallest_dataset*config.testprop)
len_smallest_trainset = len_smallest_dataset - len_smallest_testset
# name_list = ['bird', 'cat', 'chicken', 'dog', 'duck', 'gpig', 'human', 'mammal', 'man', 'mouse', 'quail', 'rabbit', 'rat', 'woman']
directory = Path(config.dataset)
num_tasks = len(list(directory.iterdir()))
tasks = [None] * num_tasks

print("filenames: ")
for filepath in directory.iterdir():
    print(filepath.stem)

for task_id, filepath in enumerate(directory.iterdir()):
    # import data
    data = pd.read_csv(filepath)

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
    train_sampler = RoundRobinBatchSampler(training_dataset, len_smallest_trainset)
    # test_sampler = RoundRobinBatchSampler(training_dataset, 25)
    train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_sampler=train_sampler, worker_init_fn=seed_worker, generator=gen, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 128, worker_init_fn=seed_worker, generator=gen, shuffle=False)

    # save DataLoader
    taskname = filepath.stem
    tasks[task_id] = (taskname, train_dataloader, test_dataloader)

tasknames = [task[0].upper() for task in tasks]

# ========================================================================================================================

# Initialize optimizer
optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config['lr'])
# Timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


# initialize helper variables
# prepare performance dataframe
performance_df = pd.DataFrame(index=tasknames + ["total"], 
                              columns=["task",
                                       "best_avg_ep", "best_avg_ep_tloss", "best_avg_ep_vloss", "best_avg_ep_AUC",
                                       "best_own_ep", "best_own_ep_tloss", "best_own_ep_vloss", "best_own_ep_AUC", ])
performance_df = performance_df.fillna(0.0)
performance_df = performance_df.infer_objects() 
performance_df["best_avg_ep"] = performance_df["best_avg_ep"].astype(int)
performance_df["best_own_ep"] = performance_df["best_own_ep"].astype(int)
performance_df["task"] = tasknames + ["total"]


early_stop = 20
stop_crit = 0
loss_fn = nn.CrossEntropyLoss()


for epoch in tqdm(range(config.epochs)):
    wandb.log({'epoch': epoch})
    # training
    if True:
        train_running_losses = [0] * num_tasks
        # zip train_dataloaders of all tasks to iterate through them in parallel
        zipped_train_dataloaders = zip(*(task[1] for task in tasks))
        # loop through batches (ith minibatch of every task)
        for i, batch in enumerate(zipped_train_dataloaders):
            train_batch_losses = [0] * num_tasks
            # loop through the tasks
            for task_id, minibatch in enumerate(batch):
                try:
                    # pass through Molformer
                    input_ids = minibatch["input_ids"]
                    attention_mask = minibatch["attention_mask"]
                    y_regression_values = minibatch["y_regression_values"]
                    with torch.no_grad():
                        outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        encoder = outputs["hidden_states"][-1]
                    # average over second dimension of encoder output to get a single vector for each example
                    encoder = encoder.mean(dim=1)
                    # pass through our model
                    preds = nnmodel(encoder, task_id) 
                    loss = loss_fn(preds, y_regression_values)

                    train_batch_losses[task_id] = loss
                    train_running_losses[task_id] += loss
                except:
                    print(f"task_id: {task_id}, batch num: {i}")
                
                # if task_id < len(batch)-1:
                #     next_minibatch_size = len(batch[task_id+1]) 
                #     if next_minibatch_size == 0:
                #         break
                    

            # unweighted
            train_batch_total_loss = sum(train_batch_losses)

            optimizer.zero_grad()
            train_batch_total_loss.backward()
            optimizer.step()

            # wandb.log({'train batch total loss': train_batch_total_loss})
        
        num_train_batches = len(tasks[0][1])
        train_avg_losses = [loss / num_train_batches for loss in train_running_losses]
        train_epoch_total_loss = sum(train_avg_losses)
        wandb.log({'train epoch total loss': train_epoch_total_loss})
        # log train loss of every task
        if num_tasks > 1:
            for task_id, taskname in enumerate(tasknames):
                wandb.log({f'train {taskname} loss': train_avg_losses[task_id]})

    # validation
    if True:
        val_dataloaders = [task[2] for task in tasks]
        val_preds = [[] for _ in range(num_tasks)]
        val_labels = [[] for _ in range(num_tasks)]
        val_running_losses = [0] * num_tasks
        val_avg_losses = [0] * num_tasks
        for task_id, dataloader in enumerate(val_dataloaders):
            num_val_minibatches = len(dataloader)
            val_running_loss = 0
            for minibatch in dataloader:
                input_ids = minibatch["input_ids"]
                attention_mask = minibatch["attention_mask"]
                y_regression_values = minibatch["y_regression_values"]
                with torch.no_grad():
                    outputs = LLModel(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    encoder = outputs["hidden_states"][-1]
                    encoder = encoder.mean(dim=1)

                    preds  = nnmodel(encoder, task_id)
                    loss = loss_fn(preds, y_regression_values)

                    val_preds[task_id].extend(preds.cpu().numpy())
                    val_labels[task_id].extend(y_regression_values.cpu().numpy())

                    val_running_loss += loss

            val_running_losses[task_id] = val_running_loss
            val_avg_losses[task_id] = val_running_loss / num_val_minibatches
            auc = calc_auc(val_labels[task_id], val_preds[task_id])
            # print(f"AUC: {auc}")

        
        # val_avg_losses = [loss / num_val_minibatches for loss in val_running_losses]
        val_total_loss = sum(val_avg_losses)
        wandb.log({'val total loss': val_total_loss})
        # log val loss of every task
        if num_tasks > 1:
            for task_id, taskname in enumerate(tasknames):
                wandb.log({f'val {taskname} loss': val_avg_losses[task_id]})

        aucs = [0] * num_tasks
        for task_id in range(num_tasks):
            auc = calc_auc(val_labels[task_id], val_preds[task_id])
            aucs[task_id] = auc

        # unweighted
        auc_avg = sum(aucs) / num_tasks

        # log auc of all 14 tasks?
        wandb.log({'val avg auc': auc_avg})
        # log auc of every task
        if num_tasks > 1:
            for task_id, taskname in enumerate(tasknames):
                wandb.log({f'val {taskname} auc': aucs[task_id]})

        # update performance_df if any aucs got better
        for task_id, taskname in enumerate(tasknames):
            if aucs[task_id] > performance_df.loc[taskname, "best_own_ep_AUC"]:
                performance_df.loc[taskname, "best_own_ep"] = epoch
                performance_df.loc[taskname, "best_own_ep_tloss"] = float(train_avg_losses[task_id])
                performance_df.loc[taskname, "best_own_ep_vloss"] = float(val_avg_losses[task_id])
                performance_df.loc[taskname, "best_own_ep_AUC"] = float(aucs[task_id])


        # take out specific tasks if their auc decreases early_stop times

        # save weights of specific last layers if their auc increases

        # early stopping and saving best results
        if auc_avg>performance_df.loc["total", "best_avg_ep_AUC"]:
            stop_crit = 0
            performance_df.loc["total", "best_avg_ep"] = epoch
            performance_df.loc["total", "best_avg_ep_tloss"] = float(train_epoch_total_loss.item())
            performance_df.loc["total", "best_avg_ep_vloss"] = float(val_total_loss.item())
            performance_df.loc["total", "best_avg_ep_AUC"] = float(auc_avg.item())

            for task_id, taskname in enumerate(tasknames):
                performance_df.loc[taskname, "best_avg_ep"] = epoch
                performance_df.loc[taskname, "best_avg_ep_tloss"] = float(train_avg_losses[task_id].item())
                performance_df.loc[taskname, "best_avg_ep_vloss"] = float(val_avg_losses[task_id].item())
                performance_df.loc[taskname, "best_avg_ep_AUC"] = float(aucs[task_id].item())

            # performance_df["best_avg_ep"] = epoch
            # performance_df["best_avg_ep_tloss"] = train_avg_losses.cpu().numpy() + [train_epoch_total_loss]
            # performance_df["best_avg_ep_vloss"] = val_avg_losses.cpu().numpy() + [val_total_loss]
            # performance_df["best_avg_ep_AUC"] = aucs + [auc_avg]


            torch.save(nnmodel.state_dict(), f'model_weights.pt')

            #model_path = 'model_{}'.format(timestamp)
            #model_scripted = torch.jit.script(nnmodel)
            #model_scripted.save(f'model_{timestamp}.pt')
            #del(model_scripted)
            #if epoch>0.75*args.epochs:
            #    # Generate Parity Plot
            #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])

            # confusion matrix
            confusion_matrices = [0] * num_tasks

            for task_id, taskname in enumerate(tasknames):
                # y_true_test = np.argmax(val_labels, axis=1)
                # y_pred_test = np.argmax(val_preds, axis=1)
                # cm = confusion_matrix(y_true_test, y_pred_test)

                y_true = np.argmax(val_labels[task_id], axis=1)
                y_pred = np.argmax(val_preds[task_id], axis=1)
                
                cm = confusion_matrix(y_true, y_pred)
                confusion_matrices[task_id] = cm
                
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'{taskname} Confusion Matrix')
                wandb.log({f"{taskname} Confusion Matrix": wandb.Image(plt)})
                plt.close()

        else:
            stop_crit+=1
        if stop_crit>early_stop:
            break

wandb.log({ "total_best_ep": performance_df.loc["total", "best_avg_ep"],
            "total_best_ep_tloss": performance_df.loc["total", "best_avg_ep_tloss"],
            "total_best_ep_vloss": performance_df.loc["total", "best_avg_ep_vloss"],
            "total_best_ep_AUC": performance_df.loc["total", "best_avg_ep_AUC"],
})

# if num_tasks > 1:
#     # log bar charts of performance for tasks and total model
#     performance_table = wandb.Table(dataframe=performance_df)
#     task_performance_table = wandb.Table(dataframe=performance_df.drop("total"))

#     # wandb.log({"best_avg_ep" : wandb.plot.bar(performance_table, "task", "best_avg_ep", title="best_avg_ep")}) # unnecessary, already in total_best_ep above, same for all tasks
#     wandb.log({"best_avg_ep_tloss" : wandb.plot.bar(task_performance_table, "task", "best_avg_ep_tloss", title="best_avg_ep_tloss")})
#     wandb.log({"best_avg_ep_vloss" : wandb.plot.bar(task_performance_table, "task", "best_avg_ep_vloss", title="best_avg_ep_vloss")})
#     wandb.log({"best_avg_ep_AUC" : wandb.plot.bar(performance_table, "task", "best_avg_ep_AUC", title="best_avg_ep_AUC")})
#     wandb.log({"best_own_ep" : wandb.plot.bar(task_performance_table, "task", "best_own_ep", title="best_own_ep")})
#     wandb.log({"best_own_ep_tloss" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_tloss", title="best_own_ep_tloss")})
#     wandb.log({"best_own_ep_vloss" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_vloss", title="best_own_ep_vloss")})
#     wandb.log({"best_own_ep_AUC" : wandb.plot.bar(task_performance_table, "task", "best_own_ep_AUC", title="best_own_ep_AUC")})

#     # columns: "best_avg_ep", "best_avg_ep_tloss", "best_avg_ep_vloss", "best_avg_ep_AUC", "best_own_ep", "best_own_ep_tloss", "best_own_ep_vloss", "best_own_ep_AUC"
#     performance_df.to_csv(f"performance{config.seed_idx}.csv", index=False)