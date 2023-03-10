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
'--beta_l2 0.5 --beta_lpips 2.0 --lpips_net vgg --batch_size 64  --runname run_2rgb_vgg_lpips2L205_B64
--beta_l2 0.5 --beta_lpips 2.0 --lpips_net squeeze --batch_size 64  --runname run_2rgb_squeeze_lpips2L205_B64
--beta_l2 0.5 --beta_lpips 2.0 --lpips_net alex --batch_size 64  --runname run_2rgb_alex_lpips2L205_B64
--beta_l2 0.5 --beta_lpips 5.0 --lpips_net vgg --batch_size 64  --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_vgg_lpips5L205_B64
--beta_l2 0.5 --beta_lpips 5.0 --lpips_net squeeze --batch_size 64  --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_squeeze_lpips5L205_B64
--beta_l2 0.5 --beta_lpips 5.0 --lpips_net alex --batch_size 64  --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_alex_lpips5L205_B64
--beta_l2 0.5 --beta_lpips 5.0 --gradclipnorm 10.0 --lpips_net vgg --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leaky_vgg_lpips5L205_gradclip
--beta_l2 0.5 --beta_lpips 5.0 --gradclipnorm 10.0 --lpips_net squeeze --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leaky_squeeze_lpips5L205_gradclip
--beta_l2 0.5 --beta_lpips 5.0 --gradclipnorm 10.0 --lpips_net alex --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leaky_alex_lpips5L205_gradclip
--beta_l2 0.5 --beta_lpips 2.0 --gradclipnorm 10.0 --lpips_net vgg --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leakyTanh_vgg_lpips2L205_gradclip
--beta_l2 0.5 --beta_lpips 2.0 --gradclipnorm 10.0 --lpips_net squeeze --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leakyTanh_squeeze_lpips2L205_gradclip
--beta_l2 0.5 --beta_lpips 2.0 --gradclipnorm 10.0 --lpips_net alex --batch_size 64 --leaky_relu_rgb 1 --blockclass Bottleneck_Up_Antichecker --runname run_antickbrd_2rgb_leakyTanh_alex_lpips2L205_gradclip
'


export cfg_param="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$cfg_param"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Repr-Inverting-GAN
python3 train_CLI.py  $cfg_param
