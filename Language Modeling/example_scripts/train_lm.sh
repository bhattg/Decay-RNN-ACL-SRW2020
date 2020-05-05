#!/bin/bash
source ../hyperparameters.txt


      CUDA_VISIBLE_DEVICES=0  python3 $lm_dir/main.py \
       --lm_data $lm_data_dir \
       --epochs $epochs \
       --model $model \
       --nhid $num_hid \
       --save $model_dir/decay_rnn_lm.pt \
       --save_lm_data $model_dir/lstm_lm.bin \
       --log-interval $log_freq \
       --batch $batch_size \
       --dropout $dropout \
       --lr $lr \
       --trainfname $train \
       --validfname $valid \
       --testfname $test \
       --save_data $save_data \
       --load_data $load_data \
       --cuda \
       --nlayers $nlayers \
       --debugger $debug \
       --load_model True


