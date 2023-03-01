#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=10-12
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o invert_resnet_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--beta_l2 0.1 --beta_lpips 1.0 --lr 0.00005 --lpips_net vgg --batch_size 64  --runname run_mod_sml_vgg_lr5-5_B64
--beta_l2 0.1 --beta_lpips 1.0 --lr 0.00005 --lpips_net squeeze --batch_size 64  --runname run_mod_sml_squeeze_lr5-5_B64
--beta_l2 0.1 --beta_lpips 1.0 --lr 0.00005 --lpips_net alex --batch_size 64  --runname run_mod_sml_alex_lr5-5_B64
'


export cfg_param="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$cfg_param"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Repr-Inverting-GAN
python3 train_modResnet_CLI.py  $cfg_param
