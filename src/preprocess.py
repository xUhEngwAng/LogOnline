import os
import pandas as pd
import re

import sys
sys.path.append("./")

from SPINE.log_parser import LogParser
from SPINE.evaluator import evaluate
        
BGL_config = {
    'head_token_num': 3,
    'delimiters': '[\s,=\[\]\$/\{\}\(\)@"\'!\+\*#\\\\;]+',
    'clustering_threshold': 0.1,
    'matching_threshold': 0.20,
    'groundtruth_path': './data/BGL/BGL_2k.log_structured.csv',
    'result_path_prefix': './data/BGL/tmp/tmp_parsed_result_node'
}

HDFS_config = {
    'head_token_num': 2,
    'delimiters': '[\s,=\[\]\$/\{\}\(\)@"\'!\+\*#\\\\;]+',
    'clustering_threshold': 0.5,
    'matching_threshold': 0.20,
    'groundtruth_path': './data/HDFS/HDFS_2k.log_structured.csv',
    'result_path_prefix': './data/HDFS/tmp/tmp_parsed_result_node'
}

def assemble_result(result_path_prefix, delete_temp=True):
    result_dir = os.path.split(result_path_prefix)[0]
    path_prefix = os.path.split(result_path_prefix)[1]
    parsed_log_df = pd.DataFrame()
    
    for filename in os.listdir(result_dir):
        if filename.startswith(path_prefix) and filename.endswith('.csv'):
            filepath = os.path.join(result_dir, filename)
            temp_parsed_df = pd.read_csv(filepath, index_col='LineId')
            parsed_log_df = pd.concat([parsed_log_df, temp_parsed_df])
            
            if delete_temp:
                os.remove(filepath)
    
    parsed_log_df = parsed_log_df.sort_index()
    return parsed_log_df

def extractBlockId(log_message):
    match = re.search('(blk_-?\\d+)', log_message)
    if not match:
        print("Fatal error, cannot extract block id for log message {}".format(log_message))
        exit(0)
    return match.group(0)

def preprocessHDFS(config, 
                   HDFS_log_path, 
                   HDFS_label_path,
                   evaluation=False):
    log_list = []
    log_file_dir = os.path.split(HDFS_log_path)[0]
    log_file_name = os.path.split(HDFS_log_path)[1].split('.')[0]
    
    with open(HDFS_log_path) as f:
        for log_line in f.readlines():
            # remove header & \n at the end of each line
            log_line = log_line.partition(": ")[2][:-1]
            log_list.append(log_line)
            
    parser = LogParser(config)
    parser.fit_parse(log_list, n_nodes=4)
    
    result_path_prefix = config.get('result_path_prefix')
    parsed_log_df = assemble_result(result_path_prefix)
    parsed_log_df['LogMessage'] = log_list
    parsed_log_df['Session'] = parsed_log_df['LogMessage'].apply(extractBlockId)
    
    if HDFS_label_path is not None:
        anomaly_label_df = pd.read_csv(HDFS_label_path)
        anomaly_label_df = anomaly_label_df.rename(columns={"BlockId": "Session", "Label": "Anomaly"})
        merged_log_df = pd.merge(parsed_log_df, anomaly_label_df, on="Session", how="left", sort=False)
        merged_log_df['Anomaly'] = merged_log_df['Anomaly'] != 'Normal'
        
    before_merge_eventId = parsed_log_df['EventId']
    after_merge_eventId = merged_log_df['EventId']
    match = [before != after for before, after in zip(before_merge_eventId, after_merge_eventId)]
    assert(sum(match) == 0)
    
    # dump parsed structured HDFS logs to file
    parsed_result_name = log_file_name + "_parsed_result.csv"
    parsed_result_path = os.path.join("./data/HDFS", parsed_result_name)
    merged_log_df.to_csv(parsed_result_path, index=False)
    
    # evaluate parsed result
    if evaluation:
        groundtruth_file_name = log_file_name + '.log_structured.csv'
        groundtruth_file_path = os.path.join(log_file_dir, groundtruth_file_name)
        if os.path.isfile(groundtruth_file_path):
            f_measure, accuracy = evaluate(groundtruth_file_path, parsed_result_path, True)
            print("F Measure: {f_measure}".format(f_measure=f_measure))
            print("Parsing Accuracy: {accuracy}".format(accuracy=accuracy))

def preprocessBGL(config,
                  BGL_log_path,
                  evaluation=False):
    log_list, labels, timestamps, nodes = [], [], [], []
    log_file_dir = os.path.split(BGL_log_path)[0]
    log_file_name = os.path.split(BGL_log_path)[1].split('.')[0]
    
    with open(BGL_log_path) as f:
        for log_line in f.readlines():
            # remove header & \n at the end of each line
            log_tokens = log_line.split()
            
            labels.append(log_tokens[0] != '-')
            timestamps.append(int(log_tokens[1]))
            nodes.append(log_tokens[3])
            
            log_line = ' '.join(log_tokens[9:])
            log_list.append(log_line)
            
    parser = LogParser(config)
    parser.fit_parse(log_list, n_nodes=4)
    
    result_path_prefix = config.get('result_path_prefix')
    parsed_log_df = assemble_result(result_path_prefix)
    
    parsed_log_df['LogMessage'] = log_list
    parsed_log_df['Anomaly'] = labels
    parsed_log_df['Timestamp'] = timestamps
    parsed_log_df['Session'] = nodes

    # dump parsed structured HDFS logs to file
    parsed_result_name = log_file_name + "_parsed_result.csv"
    parsed_result_path = os.path.join("./data/BGL", parsed_result_name)
    parsed_log_df.to_csv(parsed_result_path, index=False)
    
    # evaluate parsed result
    if evaluation:
        groundtruth_file_name = log_file_name + '.log_structured.csv'
        groundtruth_file_path = os.path.join(log_file_dir, groundtruth_file_name)
        if os.path.isfile(groundtruth_file_path):
            f_measure, accuracy = evaluate(groundtruth_file_path, parsed_result_path, True)
            print("F Measure: {f_measure}".format(f_measure=f_measure))
            print("Parsing Accuracy: {accuracy}".format(accuracy=accuracy))
            
    return parsed_log_df
    
if __name__ == '__main__':
    # HDFS_log_path = "./data/HDFS/HDFS_2k.log"
    # HDFS_anomaly_path = "./data/HDFS/anomaly_label.csv"
    # preprocessHDFS(HDFS_config, HDFS_log_path, HDFS_anomaly_path, True)
    
    BGL_log_path = './data/BGL/BGL_2k.log'
    preprocessBGL(BGL_config, BGL_log_path, False)
