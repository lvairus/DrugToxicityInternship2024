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
    

# parse arguments
parser = ArgumentParser()#add_help=False)
parser.add_argument(
    "-d", "--dataset", type=Path, required=False, help="Input data for training/validation"
)
parser.add_argument(
    "-s", "--smilescol", type=str, required=False, help="Column for SMILES"
)
parser.add_argument(
    "-l", "--labelcol", type=str, required=False, help="Column for labels"
)
parser.add_argument(
    "-t", "--testprop", type=float, required=False, help="Proportion of data used for training"
)
parser.add_argument(
    "-E", "--epochs", type=int, required=False, help="Number of epochs"
)

args = parser.parse_args()
# args, unknown_args = parser.parse_known_args()

if True:
    args.dataset = Path("gpig")
    args.smilescol = "SMILES"
    args.labelcol = "EPACategoryIndex"
    args.testprop = 0.2
    args.epochs = 10


# hyperparameters
# task_name_list = ['bird', 'cat', 'chicken', 'dog', 'duck', 'gpig', 'human', 'mammal', 'man', 'mouse', 'quail', 'rabbit', 'rat', 'woman']
task_name_list = ['bird', 'cat', 'chicken', 'dog', 'duck', 'gpig', 'mammal', 'man', 'quail', 'rabbit', 'woman']

# PROJECT = 'Multitask Class Oral Test'
PROJECT = "Multitask_Class_Oral"

sweep_config = {
    'method': 'grid',
    'metric': {'name': 'val avg auc', 'goal':'maximize'},
    'parameters': { 'seed_idx': {'values': list(range(8))},
                    'task': {'value': args.dataset},
                    'input_size': {'value': 768},
                    'emb_size': {'value': 256},
                    'hidden_size': {'value': 256},
                    'output_size': {'value': 5},
                    'lr': {'value': 1e-4},
                    'test_size': {'value': args.testprop},
                    'epochs': {'value': args.epochs},
                    'layertype': {'value': 'OrthoLinear'},
                    'smilescol': {'value': args.smilescol},
                    'labelcol': {'value': args.labelcol},
                    }
}

# sweep_id = wandb.sweep(sweep_config, project=PROJECT)

with open("sweep_id.txt", 'w') as file:
    file.write(sweep_id)

def train_model(config=None):

    # init wandb to log results
    wandb.init( project = PROJECT,
                group = "stask",
                notes = "parallel try",
                config = config,
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
    len_smallest_testset = math.ceil(len_smallest_dataset*config.test_size)
    len_smallest_trainset = len_smallest_dataset - len_smallest_testset
    directory = Path(f'single_data/{config.task[0]}')
    num_tasks = len(list(directory.iterdir()))
    tasks = [None] * num_tasks

    print("filenames: ")
    for filepath in directory.iterdir():
        print(filepath.stem)

    for task_id, filepath in enumerate(directory.iterdir()):
        # import data
        data = pd.read_csv(filepath)

        # split data
        # X_train, X_test = sklearn.model_selection.train_test_split(data[args.smilescol], test_size=args.testprop, stratify=data[args.labelcol], random_state=42)
        # Y_train, Y_test = sklearn.model_selection.train_test_split(data[args.labelcol], test_size=args.testprop, stratify=data[args.labelcol], random_state=42)
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            data[config.smilescol],
            data[config.labelcol],
            test_size=config.test_size,
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
        filename = filepath.stem
        tasks[task_id] = (filename, train_dataloader, test_dataloader)


    # ========================================================================================================================

    # Initialize optimizer
    optimizer = torch.optim.Adam(nnmodel.parameters(), lr=config.lr)
    # Timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # initialize helper variables
    early_stop = 20
    stop_crit = 0
    best_epoch_auc = 0
    loss_fn = nn.CrossEntropyLoss()

    # for i, batch in enumerate(zip(*(task[1] for task in tasks))):
    #     try:
    #         print(i)
    #         print(len(batch))
    #     except:
    #         print(f"batch num: {i}")


    for epoch in tqdm(range(config.epochs)):
        wandb.log({'epoch': epoch})
        # training
        # loop through batches (ith minibatch of every task)
        if True:
            train_running_losses = [0] * num_tasks
            # zip train_dataloaders of all tasks to iterate through them in parallel
            zipped_train_dataloaders = zip(*(task[1] for task in tasks))
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

                # log loss of each 14 tasks?
                # wandb.log({'train batch total loss': train_batch_total_loss})
            
            num_train_batches = len(tasks[0][1])
            train_avg_losses = [loss / num_train_batches for loss in train_running_losses]
            train_epoch_total_loss = sum(train_avg_losses)
            wandb.log({'train epoch total loss': train_epoch_total_loss})

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
                print(f"AUC: {auc}")

            
            val_avg_losses = [loss / num_val_minibatches for loss in val_running_losses]
            val_total_loss = sum(val_avg_losses)
            # log val loss of all 14 tasks?
            wandb.log({'val total loss': val_total_loss})

            aucs = [0] * num_tasks
            for task_id in range(num_tasks):
                auc = calc_auc(val_labels[task_id], val_preds[task_id])
                aucs[task_id] = auc

            # unweighted
            auc_avg = sum(aucs) / num_tasks

            # log auc of all 14 tasks?
            wandb.log({'val avg auc': auc_avg})

            # take out specific tasks if their auc decreases early_stop times

            # save weights of specific last layers if their auc increases

            # early stopping and saving best results
            if auc_avg>best_epoch_auc:
                stop_crit = 0
                best_epoch_auc = auc_avg
                best_epoch = epoch
                best_epoch_tloss = train_epoch_total_loss
                best_epoch_vloss = val_total_loss

                torch.save(nnmodel.state_dict(), f'model_weights.pt')

                #model_path = 'model_{}'.format(timestamp)
                #model_scripted = torch.jit.script(nnmodel)
                #model_scripted.save(f'model_{timestamp}.pt')
                #del(model_scripted)
                #if epoch>0.75*args.epochs:
                #    # Generate Parity Plot
                #    generate_parity_plot(outputs_dict["ground_truth"], outputs_dict["predictions"])

                # confusion matrix? average...? aggregate?
                # y_true_test_rat = np.argmax(val_labels_rat, axis=1)
                # y_pred_test_rat = np.argmax(val_preds_rat, axis=1)
                # cm_rat = confusion_matrix(y_true_test, y_pred_test)
                # plt.figure(figsize=(10, 7))
                # sns.heatmap(cm_rat, annot=True, fmt='d', cmap='Blues')
                # wandb.log({"RAT Confusion Matrix": wandb.Image(plt)})

            else:
                stop_crit+=1
            if stop_crit>early_stop:
                break

    wandb.log({ "best epoch": best_epoch,
                "best epoch auc": best_epoch_auc,
                "best epoch tloss": best_epoch_tloss,
                "best epoch vloss": best_epoch_vloss
    })


# start sweep
wandb.agent(sweep_id, train_model)
