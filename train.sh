GPU_ID=$1

MODEL_FILE='gpu'$GPU_ID'.'$RNN'.dim'$DIM

DATA_DIR=$HOME'/chl/robo5/data'
SAVE_DIR=$HOME'/chl/robo5/results'

TR_FILE='classification.tr'
VAL_FILE='classification.val'

CUDA_VISIBLE_DEVICES=$GPU_ID,CUDA_LAUNCH_BLCOKING=1 python3 main.py \
    --savePath=$MODEL_FILE
