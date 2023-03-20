export CUDA_VISIBLE_DEVICES=0
nohup python ./src/train.py ./data/BGL/BGL_parsed_result_03.csv \
--filter_abnormal \
--model loganomaly \
--n_epoch 50 \
--online_mode \
--partition_method timestamp \
--session_size 200 \
--top_k 40 \
--window_size 10 \
>> ./log/bgl_timestamp_200_02/loganomaly_online.log 2>&1 &
