#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --job-name=twins_dump
#SBATCH --output=log/twins_dump_%j.out

# The generated values of trained rule-twin checkpoints on the unseen puzzles 6300..7299, for analysis/rule_orbits.py
# --dumps: are rows whose first step no training row had answered right (docs/ARITHMETIC.md, "Sum and midpoint compared")?
#   sbatch jobs/pynl/submit_twins_dump.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

OUT=experiments/twins_eval
mkdir -p ${OUT}
for run in ${RUNS:-shared_midpoint_n100 shared_midpoint_n300 shared_midpoint_n600 shared_mirror_n100 shared_mirror_n600 pervalue_midpoint_n100 pervalue_midpoint_n600}; do
  rule=$(echo ${run} | cut -d_ -f2)
  case "${run}" in
    shared_*)   FLAGS="--pair --kernel 50 --ring" ;;
    lplus_*)    FLAGS="--pair --kernel 50 --ring" ;;
    pervalue_*) FLAGS="--pair --kernel 50" ;;
  esac
  python3 -u core/pynl/value_axis.py --iravenx data/iravenx_twins/${rule}/center_single_shuffle_n_3_maxval_50.json ${FLAGS} \
      --eval experiments/twins/${run} --dump ${OUT}/${run}.json --n-val 1000 --jvm-gb 32 2>&1 | grep -v "^WARNING\|SLF4J"
done
