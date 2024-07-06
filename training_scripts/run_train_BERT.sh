#!/bin/bash

#SBATCH --job-name=BERT-BNC
#SBATCH --output=output_files/BERT/XLM-RoBERTa-BNC.txt
#SBATCH --ntasks=1 # assumably different if you want multiple tasks?
#SBATCH --time=72:00:00
#SBTACH --mail-user=innes@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 -u LLM-aspect/training_scripts/train_BERT.py