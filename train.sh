GPU_ID=$1
EXP_NAME='hello'

DATA_DIR=$HOME'/chl/robo5/data'
SAVE_DIR=$HOME'/chl/robo5/result'

MODEL_FILE=$SAVE_DIR'/gpu'$GPU_ID'.'$EXP_NAME'.pth'

TR_FILE=$DATA_DIR'classification.tr'
VAL_FILE=$DATA_DIR'classification.val'

CUDA_VISIBLE_DEVICES=$GPU_ID,CUDA_LAUNCH_BLCOKING=1 python3 main.py \
    --savePath=$MODEL_FILE
