root_path_name=./dataset/
data_path_name=ETTm1.csv
data_name=ETTm1
seq_len=96

for pred_len in 96
do
for random_seed in 2024 2025 2026
do
    # A0: Original GTR baseline
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id GTR_ETTm1Ablation'_'$seq_len'_'$pred_len'_A0_gtr_base' \
      --model GTR \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --train_epochs 3 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed

    # A1: DGTR full (dynamic + multi-branch + cross-var)
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id DGTR_ETTm1Ablation'_'$seq_len'_'$pred_len'_A1_full' \
      --model DGTR \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --dgtr_spectrum_k 4 \
      --dgtr_freq_smooth 0.2 \
      --dgtr_branch_kernels 3,7,15,31 \
      --dgtr_gate_temp 1.0 \
      --dgtr_use_multiscale 1 \
      --dgtr_use_cross_var_mixer 1 \
      --dgtr_use_agg 1 \
      --train_epochs 3 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed

    # A2: DGTR without dynamic cycle (single base period only)
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id DGTR_ETTm1Ablation'_'$seq_len'_'$pred_len'_A2_noDynamic' \
      --model DGTR \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --dgtr_spectrum_k 4 \
      --dgtr_freq_smooth 0.2 \
      --dgtr_branch_kernels 3,7,15,31 \
      --dgtr_gate_temp 1.0 \
      --dgtr_use_multiscale 0 \
      --dgtr_use_cross_var_mixer 1 \
      --dgtr_use_agg 1 \
      --train_epochs 3 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed

    # A3: DGTR single branch kernel (no multi-branch fusion)
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id DGTR_ETTm1Ablation'_'$seq_len'_'$pred_len'_A3_noMultiBranch' \
      --model DGTR \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --dgtr_spectrum_k 4 \
      --dgtr_freq_smooth 0.2 \
      --dgtr_branch_kernels 31 \
      --dgtr_gate_temp 1.0 \
      --dgtr_use_multiscale 1 \
      --dgtr_use_cross_var_mixer 1 \
      --dgtr_use_agg 1 \
      --train_epochs 3 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed

    # A4: DGTR without cross-variable enhancement (no mixer + no agg)
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id DGTR_ETTm1Ablation'_'$seq_len'_'$pred_len'_A4_noCrossVar' \
      --model DGTR \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --dgtr_spectrum_k 4 \
      --dgtr_freq_smooth 0.2 \
      --dgtr_branch_kernels 3,7,15,31 \
      --dgtr_gate_temp 1.0 \
      --dgtr_use_multiscale 1 \
      --dgtr_use_cross_var_mixer 0 \
      --dgtr_use_agg 0 \
      --train_epochs 3 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

