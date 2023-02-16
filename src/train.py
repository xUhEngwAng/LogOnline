import argparse
import logging
import pandas as pd
import torch

from dataset import LogDataset
from model import DeepLog, LogAnomaly, UniLog
from partition import partition
from torch.utils.data import DataLoader
from vocab import buildVocab

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def collate_fn(batch_input_dict):
    keys = [input_dict['session_key'] for input_dict in batch_input_dict]
    templates = [input_dict['templates'] for input_dict in batch_input_dict]
    event_ids = [input_dict['eventids'] for input_dict in batch_input_dict]
    next_logs = [input_dict['next'] for input_dict in batch_input_dict]
    anomaly = [input_dict['anomaly'] for input_dict in batch_input_dict]
    
    return {'session_key': keys,
            'templates': templates,
            'eventids': event_ids,
            'next': next_logs,
            'anomaly': anomaly}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path', type=str, help='path of structured log file')
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--eval_batch_size', default=None, type=int)
    parser.add_argument('--input_size', default=64, type=int)
    parser.add_argument('--hidden_size', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--pretrain_path', default=None, type=str, help='path of pretrained word embeddings')
    parser.add_argument('--session_size', default=100, type=int)
    parser.add_argument('--step_size', default=1, type=int)
    parser.add_argument('--window_size', default=10, type=int)
    
    parser.add_argument('--filter_abnormal', action='store_true')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model', default='deeplog', type=str)
    parser.add_argument('--n_epoch', default=300, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--partition_method', default='session', type=str)
    parser.add_argument('--reset_enabled', action='store_true')
    parser.add_argument('--top_k', default=9, type=int)
    parser.add_argument('--train_ratio', default=0.01, type=float)
    
    args = parser.parse_args()
    
    # Load Pretrained word embeddings
    if args.model == 'loganomaly' and args.pretrain_path is None:
        logger.error(f'Fatal error. pretrain_path must be specified when running {args.model}.')
        exit(0)
        
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using {device} device.")
    
    parsed_log_df = pd.read_csv(args.path)
    parsed_log_df.fillna({'Templates': ''}, inplace=True)
    logger.info(f'Loading structured log file from {args.path}, {len(parsed_log_df)} log messages loaded.')
        
    if args.model == 'loganomaly' and args.pretrain_path is not None:
        args.input_size = 300
        embedding_matrix = buildVocab(parsed_log_df, args.pretrain_path)
                
    # Dataset Preparation
    session_train, session_test, num_classes = partition(parsed_log_df, 
                                                         args.partition_method, 
                                                         args.train_ratio, 
                                                         args.filter_abnormal,
                                                         args.session_size)
    
    # Obtain unique templates of the training data
    eventid_templates = {}
    
    for ind, event_id in enumerate(parsed_log_df['EventId']):
        if num_classes <= event_id:
            continue
        eventid_templates.setdefault(event_id, parsed_log_df['Templates'][ind])
        
    training_uniq_templates = list(eventid_templates.values())
    logger.info(f'{len(training_uniq_templates)} unique templates identified in training data.')
    assert(len(training_uniq_templates) == num_classes)
    
    dataset_train = LogDataset(session_train, 
                               args.window_size, 
                               args.step_size, 
                               num_classes)
    
    dataset_test = LogDataset(session_test, 
                              args.window_size, 
                              args.step_size, 
                              num_classes)
    
    logger.info(f'Successfully loaded training dataset, which has {len(dataset_train)} instances.')
    logger.info(f'Successfully loaded testing dataset, which has {len(dataset_test)} instances.')
    
    dataloader_train = DataLoader(dataset_train, 
                                  collate_fn=collate_fn, 
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  pin_memory=False)
    
    dataloader_test = DataLoader(dataset_test, 
                                 collate_fn=collate_fn, 
                                 batch_size=args.eval_batch_size, 
                                 shuffle=False, 
                                 pin_memory=False)
    
    if args.model == 'deeplog':
        model = DeepLog(num_classes,
                        args.num_layers, 
                        args.input_size, 
                        args.hidden_size, 
                        args.top_k).to(device)
        
    elif args.model == 'loganomaly':
        model = LogAnomaly(num_classes,
                           args.num_layers, 
                           args.input_size, 
                           args.hidden_size, 
                           args.top_k,
                           embedding_matrix,
                           training_uniq_templates).to(device)
    elif args.model == 'unilog':
        model = UniLog(num_classes,
                       args.num_layers,
                       args.input_size,
                       args.hidden_size,
                       args.reset_enabled,
                       args.top_k).to(device)
    else:
        logger.error(f'Unrecognised model {args.model}, exiting...')
        exit(0)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.set_optimizer(optimizer)
    model.fit_evaluate(dataloader_train, dataloader_test, args.n_epoch)
