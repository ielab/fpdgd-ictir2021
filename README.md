# Effective and Privacy-preserving Federated Online Learning to Rank

This repo contains the code used to run experiments for the paper 'Effective and Privacy-preserving Federated Online Learning to Rank', accepted by ICTIR 2021.

Here are few steps to reproduce our experiments.

## Setup python environment
Create a conda environment for running this code using the code below.

````
conda create --name federated python=3.6
source activate federated
# assuming you want to checkout the repo in the current directory
git clone https://github.com/ielab/fpdgd-ictir2021.git && cd fpdgd-ictir2021
pip install -r requirements.txt 
````

## Download datasets
In the paper, two datasets are used, MQ2007, and MSLR-WEB10K.
- MQ2007 can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/). 
- MSLR-WEB10K can be downloaded from the Microsoft Research [website](https://www.microsoft.com/en-us/research/project/mslr/).  

After downloading data files, they have to be unpacked within the `./datasets` folder.

## Reproducing results
To reproduce our experiments result, set up corresponding parameters and run file `./runs/run_fpdgd.py`
```
python run_fpdgd.py
```
or
```
sh run_fpdgd.sh
```

To reproduce the FOLtR_ES baseline, check [FOLtR-ES](https://github.com/facebookresearch/foltr-es).
To reproduce the PDGD baseline, check [PDGD](https://github.com/HarrieO/OnlineLearningToRank).
