#!/bin/bash

#SBATCH --job-name=get_a1_ambig_entropies
#SBATCH --output=output_files/BERT/get_a1_ambig_entropies_BNC2.txt
#SBATCH --ntasks=1 # assumably different if you want multiple tasks?
#SBATCH --time=72:00:00
#SBTACH --mail-user=innes@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=45G
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 -u LLM-aspect/test_scripts/test_BERT.py