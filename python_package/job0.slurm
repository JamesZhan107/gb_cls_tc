#!/bin/bash
#SBATCH --job-name=kp_mul
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=matrix2
#SBATCH --output=log
export MODEL_INFERENCE_PATH='checkpoint_test'
export IMAGE_TRAIN_INPUT_PATH='/home/lilei/dataset/rubbish/pics_ch/'
source ~/.zshrc_conda2

echo -e "
********************************************************************
Job Name:$SLURM_JOB_NAME,Job ID:$SLURM_JOBID,Allocate Nodes:$SLURM_JOB_NODELIST
********************************************************************\n\n"
python model.py 
