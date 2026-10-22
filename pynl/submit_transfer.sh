#!/bin/bash -l
#SBATCH --partition amdfast
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --array=1-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=underscoreaskarin@gmail.com
#SBATCH --job-name=transfer
#SBATCH --output=log/slurm_transfer_%x_%A_%a.out
#SBATCH --error=log/slurm_transfer_%x_%A_%a.err

# Train small, test big (pynl_transfer.py): the primitives template trained on I-RAVEN with XML
# labels, tested unchanged on I-RAVEN-X. Arm 1: one weight per primitive shared by all attributes;
# arm 2: one weight per (attribute, primitive), as pynl_symbolic's prims model.
#   a=$(sbatch --parsable pynl/submit_transfer.sh)
#   sbatch --dependency=afterok:$a --export=ALL,TEST=1 pynl/submit_transfer.sh    # scores each checkpoint once
# I-RAVEN-X files (IBM's raven-large-language-models, src/datasets/generation/iravenx_task.py;
# --nconf defaults to 1, so pass 0 for the clean files):
#   python iravenx_task.py --n 3 --maxval 10 --nconf 0 --save_dir <repo>/data/iravenx
#   python iravenx_task.py --n 10 --maxval 1000 --nconf 0 --save_dir <repo>/data/iravenx
#   python iravenx_task.py --n 10 --maxval 1000 --nconf 10 --save_dir <repo>/data/iravenx \
#          --load_file <repo>/data/iravenx/center_single_shuffle_n_10_maxval_1000.json      (same for n 3)

echo_green() { echo -e "\033[0;32m [$(date)] $1\033[0m"; }
export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${JAVA_HOME}/lib/server:${LD_LIBRARY_PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
mkdir -p log

case ${SLURM_ARRAY_TASK_ID:-1} in
  1) W=shared ;;
  2) W=per_attr ;;
esac
D=data/iravenx/center_single_shuffle
IRX=${IRX:-${D}_n_3_maxval_10.json,${D}_n_3_maxval_10_nconf_10.json,${D}_n_10_maxval_1000.json,${D}_n_10_maxval_1000_nconf_10.json}
for f in ${IRX//,/ }; do
  [ -f "$f" ] || { echo "missing I-RAVEN-X file $f (see the header for how to make it)"; exit 1; }
done

if [ -n "${TEST}" ]; then
  echo_green "node $(hostname) — transfer ${W}: test (I-RAVEN test + I-RAVEN-X)"
  srun python3 -u pynl/pynl_transfer.py test --weights ${W} --iravenx ${IRX} --jvm-gb 48
  exit
fi
echo_green "node $(hostname) — transfer ${W}: train on I-RAVEN, 300/constellation"
srun python3 -u pynl/pynl_transfer.py check --weights ${W} --iravenx ${IRX} --n-check 20 --jvm-gb 48
srun python3 -u pynl/pynl_transfer.py train --weights ${W} --n-train 300 --n-val 300 --epochs 30 \
    --lr 0.01 --jvm-gb 48
echo_green "done"
