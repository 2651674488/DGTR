model_name=DGTR

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weatherDynamicOnly
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len'_dynamic_only' \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --dgtr_use_multiscale 1 \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done

