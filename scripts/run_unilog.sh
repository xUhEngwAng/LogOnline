# dataset=$1
# path=./data/$dataset/$dataset\_parsed_result_03.csv
# partition_method=session
# if [$dataset -eq HDFS]; then
#    partition_method=session
# fi

export CUDA_VISIBLE_DEVICES=1
nohup python ./src/train.py ./data/BGL/BGL.log_structured.csv \
--embedding_method combined \
--filter_abnormal \
--model unilog \
--n_epoch 100 \
--partition_method timestamp \
--session_size 200 \
--top_k 150 \
--train_ratio 0.8 \
--window_size 10 \
>> ./log/bgl_timestamp_200_08/unilog_combined.log 2>&1 &
