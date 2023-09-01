#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
module load StdEnv/2020
module load gcc/9.3.0
module load opencv/4.8.0
source venv/bin/activate
python scripts/segmentation_train.py --data_dir data/MICCAI_BraTS2020_TrainingData  --out_dir output --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8