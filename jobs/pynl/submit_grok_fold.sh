#!/bin/bash -l
#SBATCH --partition amdlong
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=grok_fold
#SBATCH --output=log/grok_fold_%x_%j.out

# Does the row fold grok Arithmetic? The grokking papers' setting (docs/ARITHMETIC.md, "Does it grok?"): a training set
# small enough to memorise, many updates, decoupled weight decay (AdamW-style, Power et al. 2022, Nanda et al. 2023), and
# larger initial weights (Omnigrok, Liu et al. 2023, whose MSE runs grok with them). The row fold (--columns fold,
# 305k weights on 3x10) on 3x10 Arithmetic-only puzzles, values 0..50 (data_prep/arith_big_data.py); 300 selection and 1000
# unseen puzzles of the same kind follow the training ones. One update is one puzzle, lr 3e-4.
#   for r in n1000_wd0 n1000_wd1 n1000_wd1_init3 n300_wd1; do sbatch -J $r --export=ALL,RUN=$r jobs/pynl/submit_grok_fold.sh; done

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

case "${RUN}" in
  n1000_wd0)       ARGS="--n 1000 --epochs 400 --eval-every 5000 --wd 0" ;;
  n1000_wd1)       ARGS="--n 1000 --epochs 400 --eval-every 5000 --wd 1" ;;
  n1000_wd1_init3) ARGS="--n 1000 --epochs 400 --eval-every 5000 --wd 1 --init-scale 3" ;;
  n300_wd1)        ARGS="--n 300 --epochs 1000 --eval-every 3000 --wd 1" ;;
  *) echo "set RUN to n1000_wd0, n1000_wd1, n1000_wd1_init3 or n300_wd1"; exit 1 ;;
esac
SAVE=experiments/grok_fold_${RUN}
echo "node $(hostname), run ${RUN}, saving to ${SAVE}"
python3 -u core/pynl/facet_stream.py --columns fold --widths 32,64,128 --decay-every 100 --jvm-gb 64 ${ARGS} \
    --save-dir ${SAVE} --resume 2>&1 | grep --line-buffered -v "^WARNING\|SLF4J"
