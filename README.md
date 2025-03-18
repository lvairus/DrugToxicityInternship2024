# Overview:
         
My overall goal throughout these projects was to build and test models to predict drug toxicity. Drug toxicity is one of the most common causes for failure in drug discovery pipelines, which is especially costly if it’s only found out in the later stages. Currently the most common methods to assess toxicity involve animal testing, which is expensive, time-consuming, and ethically questionable ([B. Indira Priyadarshini, et al.](https://www.powertechjournal.com/index.php/journal/article/view/322/265)). Machine learning would be a much better alternative. However, the current models aren’t very accurate, so I worked towards making models with improved performance so that they can be added to drug discovery workflows. There are many challenges which make drug toxicity a difficult task to predict. There is a broad range of ways a compound can be toxic, and the degree of toxicity depends heavily on where it is in the body and what organs or proteins it’s interacting with. It is also challenging to gather raw experimental data on toxicity since they require in-vitro/vivo experiments, and then it’s hard to get access to that data. And even with that data it’s difficult to translate non-human models to humans due to biological differences. 

# Summary:

-   What I've done
    -   Note: all of the data I used is from [TOXRIC](https://toxric.bioinforai.tech/home)
    -   Made [summary statistics](https://docs.google.com/spreadsheets/d/1xg5nN9Y46C4maEhRqN5-vszqrxwEvvPQzq6InZ-1PXQ/edit?gid=0#gid=0) of the TOXRIC datasets
    -   Experimented with different SST models
        -   No good results (R2 \< 0, AUC \< .70)
        -   Switched to using MolFormer the rest of the time
    -   Attempted to use ConPlex
        -   Binding scores were always low and I didn't find a correlation between binding score and toxicity
    -   Compared single-task vs multi-task models for all animal LD50 oral datasets
        -   [AUC summary results](https://docs.google.com/spreadsheets/d/1bLnZlA0Uy6Qdars4IBzdzGZXdK2MjRNz_X3VrXSSB0U/edit?gid=0#gid=0)
        -   [s/m-task wandb project](https://wandb.ai/lvairusorg/Multitask_Class_Oral/workspace?nw=nwuserlvairus)
    -   Compared many different models on mouse LD50 oral data to see which architecture performed the best
        -   [mouse wandb project](https://wandb.ai/lvairusorg/Mouse/workspace) (see sweeps to get performance of each model type across 10 random seeds)
    -   Ran quick optimization sweeps on multiple TOXRIC datasets
        -   [TOXRIC wandb project](https://wandb.ai/lvairusorg/Toxric/workspace) (see sweeps to get hyperparameter optimization runs for each dataset)
-   Future Directions
    -   Transfer learning
        -   Start with oral mouse to oral human data, as we currently have the most mouse data (20,000 data points)
        -   Possibly add rat data since there's a good amount (10,000 data points)
    -   Getting more data
        -   One of the reasons my models weren't doing that well could be because I didn't have enough data
    -   Using docking scores to aid machine learning
        -   Either add it as an input to a model or make a multitask model that also predicts docking score

A pdf of my full write-up for my work can be found as Toxicity_Internship_Report in this repository
