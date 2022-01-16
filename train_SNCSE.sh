#!/bin/sh

cd SNCSE

python train_SNCSE.py
--model_name_or_path bert-base-uncased
--train_file data/wiki1m_for_simcse.txt
--output_dir ../chechpoint
 --num_train_epochs 1
 --per_device_train_batch_size 256
 --learning_rate 1e-5
 --max_seq_length 32
 --metric_for_best_model stsb_spearman
 --pooler_type cls
 --mlp_only_train
 --temp 0.05
 --save_steps 125
 --do_train


python train_SNCSE.py
--model_name_or_path bert-large-uncased
--train_file data/wiki1m_for_simcse.txt
--output_dir ../chechpoint
 --num_train_epochs 1
 --per_device_train_batch_size 128
 --learning_rate 5e-6
 --max_seq_length 32
 --metric_for_best_model stsb_spearman
 --pooler_type cls
 --mlp_only_train
 --temp 0.05
 --save_steps 125
 --do_train

python train_SNCSE.py
--model_name_or_path roberta-base
--train_file data/wiki1m_for_simcse.txt
--output_dir ../chechpoint
 --num_train_epochs 1
 --per_device_train_batch_size 256
 --learning_rate 1e-5
 --max_seq_length 32
 --metric_for_best_model stsb_spearman
 --pooler_type cls
 --mlp_only_train
 --temp 0.05
 --save_steps 125
 --do_train

python train_SNCSE.py
--model_name_or_path roberta-large
--train_file data/wiki1m_for_simcse.txt
--output_dir ../chechpoint
 --num_train_epochs 1
 --per_device_train_batch_size 128
 --learning_rate 5e-6
 --max_seq_length 32
 --metric_for_best_model stsb_spearman
 --pooler_type cls
 --mlp_only_train
 --temp 0.05
 --save_steps 125
 --do_train