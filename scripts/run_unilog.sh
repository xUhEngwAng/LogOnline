CUDA_VISIBLE_DEVICES=3
python ./src/train.py \
       ./data/BGL/BGL_parsed_result_03.csv \
       --eval_batch_size 1024 \
       --filter_abnormal \
       --model unilog \
       --n_epoch 30 \
       --top_k 20 \