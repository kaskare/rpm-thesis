#!/bin/bash -l
#SBATCH --partition amdgpuextralong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=21-00:00:00
#SBATCH --job-name=grok_gnn
#SBATCH --output=log/grok_gnn_%j.out

# Will a big GNN grok Arithmetic? Attribute GNN B at 16x (25.6M weights) on 100,000 Arithmetic puzzles (data_prep/arith_big_data.py),
# with the recipe of submit_attr_gnn.sh except: a constant learning rate, so the decay never fades, and no dropout, so weight decay is
# the only regularizer. Weight decay 0.1 (it had 0.5 under a cosine schedule, which decays 7 e-folds in all; 0.1 at a constant rate
# decays 0.16 per epoch of 100,000 puzzles). Tried first, with dropout 0.3: decay 1 collapses (loss 0.022 -> 0.059 by epoch 25),
# and at 0.1-0.5 exact swings between evaluations (0.1: 30, 71, 30). 64x (102M) does not start at this learning rate: loss flat
# at 0.170 for 20 epochs. An epoch takes about 5 minutes. Resubmit to continue from last.pt.
#   sbatch jobs/baselines/submit_arith_big.sh

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python core/baselines/train_baseline.py --data iravenx_arith --arch attr-rel --seed 0 --scale 16 --epochs "${EPOCHS:-5500}" --eval-every 5 --batch 64 \
    --lr 1e-3 --warmup 1 --clip 1 --wd "${WD:-0.1}" --dropout "${DROP:-0}" --train-eval 1000 --select arithmetic --resume --tag grok_do${DROP:-0}_wd${WD:-0.1}
