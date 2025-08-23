model_name=AutoTimes_Llama

root_path=./dataset/sats_ds_short_coords//
model_dir=./llama

# training one model with a context length
torchrun --nnodes 1 --nproc-per-node 4 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $root_path \
  --model_id sat_240_40 \
  --model $model_name \
  --data 'sat' \
  --seq_len 240 \
  --label_len 200 \
  --token_len 40 \
  --test_seq_len 240 \
  --test_label_len 200 \
  --test_pred_len 40 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --use_amp \
  --mlp_hidden_layers 3 \
  --mlp_hidden_dim 512 \
  --mlp_activation tanh \
  --des 'Exp' \
  --use_multi_gpu \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --checkpoints './checkpoints' \
  --dropout 0.1 \
  --num_workers 0 \
  --patience 3 \
  --des 'test' \
  --loss 'MSE' \
  --lradj 'type1' \
  --weight_decay 0 \
  --test_dir 'long_term_forecast_sat_240_40_AutoTimes_Llama_sat_sl240_ll200_tl40_lr0.0001_bt256_wd0.0_hd512_hl0_cosTrue_mixTrue_None_test' \
  --test_file_name 'checkpoint.pth' \
  --visualize
