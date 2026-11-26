#!/bin/bash -l
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=recognizer
#SBATCH --output=log/recognizer_%x_%j.out

# The ViT object recognizer (core/frontend/train_vit_recognizer.py): the black-box ViT's architecture reading one panel
# and naming its objects, as perception for the facet model end to end. Three stages, chained:
#   prep     CPU, env pynl: panels and labels of 2000 training puzzles per layout, val 0..299 (selection), val
#            300..1299 (unseen) and test 0..1999 (data_prep/prepare_recognizer_data.py; about 15 GB)
#   train    GPU, env nvsa: width WIDTH (default 64, as the small black-box ViT), 30 epochs, picked on the selection panels
#   predict  GPU, env nvsa: probabilities per slot for the unseen and test panels, and perception scores
#   P=$(sbatch --parsable -p amd --export=ALL,STAGE=prep jobs/frontend/submit_recognizer.sh)
#   T=$(sbatch --parsable -p amdgpu --gres=gpu:1 --dependency=afterok:$P --export=ALL,STAGE=train jobs/frontend/submit_recognizer.sh)
#   sbatch -p amdgpu --gres=gpu:1 --dependency=afterok:$T --export=ALL,STAGE=predict jobs/frontend/submit_recognizer.sh
# The learning curve end to end: PER_LAYOUT=N trains on the first N puzzles of every layout, with EPOCHS = 30 * 2000 / N
# so the number of updates matches; only the unseen split is predicted. N = 3000 needs its own training data:
#   sbatch -p amd --export=ALL,STAGE=prep,N_TRAIN=3000,SPLITS=train,DATA=experiments/recognizer/data3000 jobs/frontend/submit_recognizer.sh
#   sbatch -p amdgpu --gres=gpu:1 --export=ALL,STAGE=train,PER_LAYOUT=300,EPOCHS=200 jobs/frontend/submit_recognizer.sh
#   sbatch -p amdgpu --gres=gpu:1 --export=ALL,STAGE=predict,PER_LAYOUT=300,PRED_SPLITS=unseen jobs/frontend/submit_recognizer.sh

source /mnt/appl/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh
cd /home/kassyask/neuro-vector-symbolic-architectures-raven
WIDTH=${WIDTH:-64}
PER_LAYOUT=${PER_LAYOUT:-0}
DATA=${DATA:-experiments/recognizer/data}
OUT=experiments/recognizer/vit${WIDTH}$([ "${PER_LAYOUT}" != 0 ] && echo "_n${PER_LAYOUT}")
echo "node $(hostname), stage ${STAGE}"
case "${STAGE}" in
  prep)    conda activate pynl; export PYTHONNOUSERSITE=1
           python3 -u data_prep/prepare_recognizer_data.py --n-train ${N_TRAIN:-2000} --splits ${SPLITS:-train,sel,unseen,test} --out ${DATA} --workers 8 2>&1 | grep -v WARNING ;;
  train)   conda activate nvsa; nvidia-smi -L
           python -u core/frontend/train_vit_recognizer.py train --width ${WIDTH} --epochs ${EPOCHS:-30} --data ${DATA} --per-layout ${PER_LAYOUT} --out ${OUT} ;;
  predict) conda activate nvsa; nvidia-smi -L
           python -u core/frontend/train_vit_recognizer.py predict --out ${OUT} --data ${DATA} --splits ${PRED_SPLITS:-unseen,test} ;;
  *) echo "set STAGE to prep, train or predict"; exit 1 ;;
esac
