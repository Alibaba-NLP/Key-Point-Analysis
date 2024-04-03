#!/bin/bash

export WANDB_DISABLED=true

WORKSPACE_DIR=/home2/xli/KPA/
DATASET_NAME=ArgKP
MAX_USED_COUNT=5
CKP_DIR=/home2/xli/KPA/KeyPointGeneration/outputs/"$DATASET_NAME"_large_MAX_USED_COUNT_"$MAX_USED_COUNT"

#cd $WORKSPACE_DIR/KeyPointGeneration && python ft_flant5.py \
#  --input_file $WORKSPACE_DIR/Datasets/datas_"$DATASET_NAME"_ft_v2_MAX_USED_COUNT_"$MAX_USED_COUNT"/instructions_train_MAX_USED_COUNT_"$MAX_USED_COUNT".json \
#  --model_path /home2/xli/models/flan-t5-large/ \
#  --output_dir $CKP_DIR \
#  --batch_size 16 \
#  --gradient_accumulation_steps 4 \
#  --lr 1e-5 \
#  --epoch 5 \
#  --weight_decay 0.01

# mkdir -p $WORKSPACE_DIR/GraphPartitioning/eval_outputs/"$DATASET_NAME"_large_MAX_USED_COUNT_"$MAX_USED_COUNT"
cd $WORKSPACE_DIR/GraphPartitioning && python eval_batch.py \
  --model_dir $CKP_DIR \
  --test_file $WORKSPACE_DIR/Datasets/datas_"$DATASET_NAME"_ft_v2_MAX_USED_COUNT_"$MAX_USED_COUNT"/instructions_test_MAX_USED_COUNT_"$MAX_USED_COUNT".json \
  --output_dir $WORKSPACE_DIR/GraphPartitioning/eval_outputs/"$DATASET_NAME"_large_MAX_USED_COUNT_"$MAX_USED_COUNT"
