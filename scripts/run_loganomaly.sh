CUDA_VISIBLE_DEVICES=1 nohup \
python ./src/train.py \
       ./data/HDFS/HDFS_parsed_result_03.csv \
       --filter_abnormal \
       --hidden_size 128 \
       --lr 0.1 \
       --model loganomaly \
       --n_epoch 30 \
       --pretrain_path ./data/wiki-news-300d-1M.vec \
       --top_k 9