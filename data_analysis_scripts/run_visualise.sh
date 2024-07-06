#!/bin/bash

#SBATCH --job-name=viz
#SBATCH --output=output_files/analysis/viz_MVs_BNC2.txt
#SBATCH --ntasks=1 # assumably different if you want multiple tasks?
#SBATCH --time=72:00:00
#SBTACH --mail-user=innes@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 /home/students/innes/ba2/LLM-aspect/data_analysis_scripts/visualise_MVs.py