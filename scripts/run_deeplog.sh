# dataset=$1
# path=./data/$dataset/$dataset\_parsed_result_03.csv

export CUDA_VISIBLE_DEVICES=0
nohup python ./src/train.py ./data/BGL/BGL_parsed_result_03.csv \
--filter_abnormal \
--model deeplog \
--n_epoch 10 \
--partition_method session \
--session_size 200 \
--top_k 40 \
--window_size 10 \
>> ./log/bgl_session/deeplog.log 2>&1 &
