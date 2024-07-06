#!/bin/bash

#SBATCH --job-name=russi
#SBATCH --output=output_files/BERT/german_vm_BNC.txt
#SBATCH --ntasks=1 # assumably different if you want multiple tasks?
#SBATCH --time=72:00:00
#SBTACH --mail-user=innes@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --qos=batch
#SBATCH --gres=gpu

#python3 LLM-aspect/inference_scripts/BERT_russian_vm.py
python3 LLM-aspect/inference_scripts/BERT_german_vm.py
#python3 LLM-aspect/inference_scripts/BERT_single_sent_inference.py
#python3 LLM-aspect/inference_scripts/inference.py
#python3 LLM-aspect/inference_scripts/single_sent_inference.py
#python3 LLM-aspect/inference_scripts/llama_inference.py