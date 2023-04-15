# user-defined parameters
dataset=Spirit
device_number=2
model=unilog
embedding_method=combined
thresh=0.001
train_ratio=8
online_mode=False

# always fixed parameters
session_size=200
window_size=10
shuffle=True

# computed parameters
path=./data/$dataset/$dataset.log\_structured.csv
autoencoder_path=./checkpoint/${dataset,,}\_ae\_0${train_ratio}_newpartition.pth
logging_file=$model\_$embedding_method\

if [ $online_mode = True ]; then
    logging_file+=\_online
fi

if [ $shuffle = True ]; then
    logging_file+=\_shuffle
fi

if [ $dataset = HDFS ]; then
    n_epoch=80
    logging_path=${dataset,,}\_newpartition\_0${train_ratio}
    online_level=session
    partition_method=session
    topk=15
else
    n_epoch=150
    logging_path=${dataset,,}\_newpartition\_$session_size\_0${train_ratio}
    online_level=log
    partition_method=timestamp
    topk=150
fi

export CUDA_VISIBLE_DEVICES=$device_number
nohup python ./src/train.py $path \
--autoencoder_path $autoencoder_path \
--embedding_method $embedding_method \
--model $model \
--n_epoch $n_epoch \
--online_level $online_level \
--online_mode \
--partition_method $partition_method \
--session_size $session_size \
--thresh $thresh \
--top_k $topk \
--train_ratio 0.$train_ratio \
--window_size $window_size \
>> ./log/$logging_path/$logging_file.log 2>&1 &
