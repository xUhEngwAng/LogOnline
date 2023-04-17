import argparse
import logging
import pandas as pd
import torch

from dataset import LogDataset
from feature import extractFeatures
from model import DeepLog, LogAnomaly, UniLog
from partition import partition
from torch.utils.data import DataLoader
from vocab import buildVocab

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path of structured log file')
    
    # size-related arguments
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--session_size', default=100, type=int)
    parser.add_argument('--step_size', default=1, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    
    # autoencoder-related arguments
    parser.add_argument('--autoencoder_path', type=str, help='the path of trained autoencoder used for online learning')
    parser.add_argument('--online_level', type=str, choices=['session', 'log'], help='whether online learning is performed on a log sequence (session) or a single line of log (log)')
    parser.add_argument('--online_mode', action='store_true')
    parser.add_argument('--thresh', type=float, help='the threshold used by autoencoder to determine whether a log sequence is normal')
    
    # other general arguments
    parser.add_argument('--embedding_method', default='context', type=str, choices=['context', 'semantics', 'combined'], help='the chosen embedding layer of UniLog model')
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--model', default='deeplog', type=str, choices=['deeplog', 'loganomaly', 'unilog'])
    parser.add_argument('--n_epoch', default=300, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--partition_method', default='session', type=str, choices=['session', 'timestamp'])
    parser.add_argument('--pretrain_path', default='./data/wiki-news-300d-1M.vec', type=str, help='path of pretrained word embeddings')
    parser.add_argument('--shuffle', action='store_true', help='shuffle before partitioning training and testing dataset, only valid when partition_method is set to timestamp')
    parser.add_argument('--top_k', default=9, type=int)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--unsupervised', action='store_true', help='unsupervised training of specified model')
    
    args = parser.parse_args()
    
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using {device} device.")
    
    parsed_log_df = pd.read_csv(args.path)
    parsed_log_df.fillna({'EventTemplate': ''}, inplace=True)
    logger.info(f'Loading structured log file from {args.path}, {len(parsed_log_df)} log messages loaded.')
    
    # Load Pretrained word embeddings
    embedding_matrix = None
    if args.model == 'loganomaly':
        if args.pretrain_path is None:
            logger.error(f'Fatal error, pretrain_path must be specified when running {args.model}.')
            exit(0)
        else:
            embedding_matrix = buildVocab(parsed_log_df, args.pretrain_path)
    if args.model == 'unilog' and args.embedding_method != 'context':
        if args.pretrain_path is None:
            logger.error(f'Fatal error, pretrain_path must be specified when running {args.model} with {args.embedding_method} embedding.')
            exit(0)
        else:
            embedding_matrix = buildVocab(parsed_log_df, args.pretrain_path)
            
    # Dataset Preparation
    session_train, session_test = partition(parsed_log_df, 
                                            args.partition_method, 
                                            args.session_size,
                                            args.shuffle,
                                            args.train_ratio)
    
    num_components, num_events, num_levels, uniq_events = extractFeatures(session_train, session_test, args.unsupervised)
    logger.info(f'Number of training and testing sessions after feature extraction are {len(session_train)} and {len(session_test)}.')
    
    # Obtain unique templates of the training data
    eventid_templates = {}
    
    for ind, event_id in enumerate(parsed_log_df['EventId']):
        event_id = uniq_events.get(event_id, event_id)
        try:
            event_id = int(event_id)
        except:
            continue
        if num_events <= event_id:
            continue
        eventid_templates.setdefault(event_id, parsed_log_df['EventTemplate'][ind])
        
    eventid_templates = {k: eventid_templates[k] for k in sorted(eventid_templates)}
    training_uniq_templates = list(eventid_templates.values())
    logger.info(f'{len(training_uniq_templates)} unique templates identified in training data.')
    
    args.embedding_matrix = embedding_matrix
    args.num_components = num_components
    args.num_events = num_events
    args.num_levels = num_levels
    args.training_tokens_id = training_uniq_templates
    
    if args.model == 'deeplog':
        logger.info(f'Initializing DeepLog model.')
        model = DeepLog(args).to(device)  
    elif args.model == 'loganomaly':
        logger.info(f'Initializing LogAnomaly model.')
        model = LogAnomaly(args).to(device)
    elif args.model == 'unilog':
        logger.info(f'Initializing UniLog model, embedding_method: {args.embedding_method}.')
        model = UniLog(args).to(device)
    else:
        logger.error(f'Fatal error, unrecognised model {args.model}.')
        exit(0)
        
    logger.info(f'num_classes: {num_events}, num_layers: {args.num_layers}, input_size: {args.input_size}, hidden_size: {args.hidden_size}, topk: {args.top_k}, optimizer: {args.optimizer}, lr: {args.lr}, train_ratio: {args.train_ratio}, window_size: {args.window_size}.')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
    model.setOptimizer(optimizer)
    model.fit_evaluate(session_train, 
                       session_test, 
                       args)
