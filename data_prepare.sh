DATA_DIR=$HOME'/chl/robo5/data'
TRAIN_FILE='robo_train.csv'
DUMMY_FILE='robo_dummy.csv'

TR_FILE='classification.tr'
VAL_FILE='classification.val'
TEST_FILE='classification.test'

python data_prepare.py --data_dir=$DATA_DIR \
        --train_file=$TRAIN_FILE --dummy_file=$DUMMY_FILE\
        --tr_out_file=$TR_FILE --val_out_file=$VAL_FILE --test_out_file=$TEST_FILE

