#!/bin/bash -l
#SBATCH --partition amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=notebook
#SBATCH --output=log/notebook_%x_%j.out

# Execute a notebook in place on a compute node (the login node kills CPU-heavy kernels), then write its HTML copy to docs/.
#   sbatch -J va_next --export=ALL,NB=notebooks/value_axis_next_arithmetic.ipynb jobs/pynl/submit_notebook.sh

export JAVA_HOME=/mnt/appl/software/Java/21.0.8
export PATH="${JAVA_HOME}/bin:${PATH}"
source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate pynl
export PYTHONNOUSERSITE=1
cd /home/kassyask/neuro-vector-symbolic-architectures-raven

echo "node $(hostname), executing ${NB}"
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "${NB}" && \
  jupyter nbconvert --to html --output-dir docs "${NB}"
