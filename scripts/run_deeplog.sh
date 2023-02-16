CUDA_VISIBLE_DEVICES=3
python ./src/train.py \
       ./data/BGL/BGL_parsed_result_03.csv \
       --eval_batch_size 1024 \
       --filter_abnormal \
       --model deeplog \
       --n_epoch 300 \
       --partition_method timestamp \
       --top_k 9 \
