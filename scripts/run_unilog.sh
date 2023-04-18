# user-defined parameters
dataset=Spirit
device_number=0
model=unilog
embedding_method=combined
thresh=0.0030
train_ratio=8
online_mode=True

# always fixed parameters
min_topk=0
session_size=200
window_size=10
shuffle=False

# computed parameters
path=./data/$dataset/$dataset.log\_structured.csv
# autoencoder_path=./checkpoint/${dataset,,}\_ae\_0${train_ratio}_newpartition.pth
autoencoder_path=./checkpoint/newpartition/${dataset,,}\_ae\_0${train_ratio}_epoch1.pth
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
    
    if [ $dataset = BGL ]; then
        topk=150
    else
        min_topk=100
        topk=250
    fi
fi

export CUDA_VISIBLE_DEVICES=$device_number
nohup python ./src/train.py $path \
--autoencoder_path $autoencoder_path \
--embedding_method $embedding_method \
--model $model \
--min_topk $min_topk \
--n_epoch $n_epoch \
--online_level $online_level \
--online_mode \
--partition_method $partition_method \
--session_size $session_size \
--thresh $thresh \
--topk $topk \
--train_ratio 0.$train_ratio \
--window_size $window_size \
>> ./log/$logging_path/$logging_file\_30.log 2>&1 &
