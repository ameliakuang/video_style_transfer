#!/bin/bash

# Define the values to loop over
EPOCHS_LIST=(5 10)
LR_LIST=(5e-4 1e-3)

for EPOCHS in "${EPOCHS_LIST[@]}"; do
  for LR in "${LR_LIST[@]}"; do
    echo "Running with epochs=$EPOCHS, lr=$LR"
    python DiffuseST_rl/run_policy_gradient.py \
      --train_content_path data2/train/frames_853913 \
      --test_content_path data2/train/frames_853913 \
      --style_path ./data/style/ \
      --output_dir ./output/frames_853913 \
      --latents_dir ./latents/frames_853913 \
      --num_epochs $EPOCHS \
      --lr $LR
  done
done
