# UniLog

## Dataset Preparation and Dependencies

To run UniLog, the dependencies should first be installed via

```
pip install -r requirements.txt
```

In this paper, all the experiments are conducted on two public log dataset, namely HDFS and BGL.
However, we do not include them in this repo due to a lack of space. 
The original HDFS and BGL dataset can be obtained from [loghub](https://github.com/logpai/loghub).
Another recent [survey](https://github.com/LogIntelligence/LogADEmpirical/tree/icse2022) on log-based anomaly detection maintained a collection of parsed log data, which can be obtained from [this website](https://figshare.com/s/8e367db4d98cf39203c5).
You are encouraged to download these dataset before running our method. The downloaded files should be placed under the `/data` folder, in the directory named after the dataset name. 
For example, the parsed HDFS file `HDFS.log_structured.csv` should be placed under `/data/HDFS/HDFS.log_structured.csv`.

## Training of the normality detection model

The normality detection model is trained separately using the notebook script in `src/autoencoder.ipynb`.
Afterwards, the parameter files of the trained normality detection model are placed under `checkpoint/` directory, which are loaded by the anomaly detection model in the evaluation phase.
For ease of use, we already placed the checkpoints for both HDFS and BGL dataset under the `checkpoint/` directory.

## Running of UniLog

You can run our proposed UniLog by the following command

```
bash scripts/run_unilog.sh
```

The log generated is written to a log file under the `/log` directory.
You can modify the shell script in order to try different experimental settings, such the dataset used, session size and window size, number of LSTM layers and so on.
Check the possible parameters by running

```
python src/train.py --help
```
