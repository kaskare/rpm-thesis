#!/bin/bash -l
#SBATCH --partition amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=bl_test
#SBATCH --output=log/baseline_test_%j.out

# The torch baselines of the main table on the I-RAVEN test split, once each (core/baselines/test_eval.py): every
# checkpoint first has to reproduce its recorded numbers on the unseen validation puzzles, then it is scored on test
# 0..1999 of every layout (14,000 puzzles, the facet model's test pass) and <run>/test.json is written. A run that has
# a test.json is skipped. The test arrays and images come first:
#   PREP=$(sbatch --parsable -p amdfast -c 8 --mem=24G -t 1:00:00 -J bl_test_data -o log/baseline_test_data_%j.out \
#     --wrap "source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh && conda activate pynl && \
#             export PYTHONNOUSERSITE=1 && cd /home/kassyask/neuro-vector-symbolic-architectures-raven && \
#             python3 data_prep/prepare_baseline_data.py --test && python3 data_prep/prepare_baseline_images.py --test")
#   sbatch --dependency=afterok:${PREP} jobs/baselines/submit_test_once.sh
#   DRY=1 sbatch jobs/baselines/submit_test_once.sh        # only the unseen check, the test split untouched
# The checkpoints: the main table's I-RAVEN rows (docs/RESULTS.md section 1). The other runs in experiments/baselines/runs
# are the sweeps those settings were picked from, on selection puzzles; they get no test pass.

R=experiments/baselines/runs
RUNS="$R/iraven_attr-rel_reg_do0_wd0.5_s0 $R/iraven_attr-abs_reg_do0_wd0.5_s0
      $R/iraven_gnn_reg_do0_wd0.5_s0 $R/iraven_gnn_reg_do0_wd0.5_s1 $R/iraven_gnn_s0 $R/iraven_gnn_s1
      $R/iraven_gnn_x16_reg_do0_wd0.5_s0
      $R/iraven_mlp_reg_do0_wd0.5_s0 $R/iraven_mlp_reg_do0_wd0.5_s1 $R/iraven_mlp_s0 $R/iraven_mlp_s1
      $R/iraven_vit_x16_wd0.05_do0.1_s0 $R/iraven_vit_wd0.05_do0.1_s0
      $R/iraven_vit-sym_x16_n6000_s0 $R/iraven_vit-sym_n6000_s0"
# TEST_RUNS replaces the list.
RUNS=${TEST_RUNS:-$RUNS}

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate nvsa
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
nvidia-smi -L
python -u core/baselines/test_eval.py ${RUNS} ${DRY:+--unseen-only}
