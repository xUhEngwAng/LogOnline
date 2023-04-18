import logging
import numpy as np
import time
import torch

from autoencoder import AutoEncoder
from dataset import LogDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def collate_fn(batch_input_dict):
    keys = [input_dict['session_key'] for input_dict in batch_input_dict]
    templates = [input_dict['templates'] for input_dict in batch_input_dict]
    event_ids = [input_dict['eventids'] for input_dict in batch_input_dict]
    elapsed_time = [input_dict['elapsedtime'] for input_dict in batch_input_dict]
    components = [input_dict['components'] for input_dict in batch_input_dict]
    levels = [input_dict['levels'] for input_dict in batch_input_dict]
    
    next_logs = [input_dict['next'] for input_dict in batch_input_dict]
    anomaly = [input_dict['anomaly'] for input_dict in batch_input_dict]
    autoencoder_pred = [input_dict['autoencoder_pred'] for input_dict in batch_input_dict]

    return {'session_key': keys,
            'templates': templates,
            'eventids': event_ids,
            'elapsedtime': elapsed_time,
            'components': components,
            'levels': levels,
            'next': next_logs,
            'anomaly': anomaly,
            'autoencoder_pred': autoencoder_pred}

class BaseModel(torch.nn.Module):
    def __init__(self, options):
        super(BaseModel, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = None
        self.online_mode = options.online_mode
        self.online_level = options.online_level
        self.thresh = options.thresh
        self.min_topk = options.min_topk
        self.topk = options.topk
        
        if self.online_mode:
            autoencoder = AutoEncoder(options.num_components, 
                                      options.num_levels, 
                                      options.window_size+1,
                                      self.thresh).to('cuda')
            autoencoder.load_state_dict(torch.load(options.autoencoder_path))
            self.autoencoder = autoencoder.eval()
            self.autoencoder_loss = torch.nn.MSELoss(reduction='none')
            logger.info(f'Successfully loaded autoencoder params from {options.autoencoder_path}.')
            logger.info(f'Online learning of {options.online_level} level is applied.')
    
    def evaluate(self, test_loader):
        if self.online_mode:
            logger.info('Online mode enabled, model would get updated during evaluation.')
            model = self.train()
        else:
            model = self.eval()
            
        batch_cnt = 0
        total_loss = 0
        session_dict = {}
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        for batch in test_loader:
            batch_cnt += 1
            
            pred = model.forward(batch)
            batch_size, num_classes = pred.shape[0], pred.shape[1]
            
            batch_next_log = [next_log['eventid'] for next_log in batch['next']]
            # the ranking of the next log is the sum of all candidates with pred score greater than it
            batch_ranking = torch.sum(torch.gt(pred.t(), pred[range(batch_size), batch_next_log]), axis=0)
            batch_ranking_list = batch_ranking.tolist()
            
            # back-propagation in online mode
            if self.online_mode:
                if self.online_level == 'log':
                    batch_embedding, output = self.autoencoder(batch)
                    autoencoder_loss = self.autoencoder_loss(output, batch_embedding).mean(axis=1)
                    weight = torch.lt(autoencoder_loss, self.thresh).to(torch.float)
                else:
                    weight = torch.tensor(batch['autoencoder_pred'], dtype=torch.float).to('cuda')
                    
                weight_sum = torch.sum(weight).item()
                label = torch.tensor(batch_next_log).to('cuda')
                batch_loss = loss(pred, label)
                batch_loss = torch.matmul(batch_loss, weight) / (weight_sum + 1e-6)
                total_loss += batch_loss

                self.optim.zero_grad()
                batch_loss.backward()
                self.optim.step()

            # Aggregate prediction results to sessions
            for ind in range(batch_size):
                session_key = batch['session_key'][ind]
                label = batch['anomaly'][ind]
                if session_key not in session_dict:
                    session_dict[session_key] = {f'matched_{topk}': True for topk in range(self.min_topk, self.topk)}
                    session_dict[session_key]['anomaly'] = False
                session_dict[session_key]['anomaly'] |= label
                
                for topk in range(self.min_topk, self.topk):
                    session_dict[session_key][f'matched_{topk}'] &= (batch_ranking_list[ind] <= topk)
                            
        TP = [0] * (self.topk - self.min_topk)
        FP = [0] * (self.topk - self.min_topk)
        TON = 0 # total negative
        TOP = 0 # total positive
        
        for key, session_info in session_dict.items():
            if session_info['anomaly']:
                TOP += 1
                for topk in range(self.min_topk, self.topk):
                    if not session_info[f'matched_{topk}']:
                        TP[topk-self.min_topk] += 1
            else:
                TON += 1
                for topk in range(self.min_topk, self.topk):
                    if not session_info[f'matched_{topk}']:
                        FP[topk-self.min_topk] += 1
                    
        logger.info(f'Evaluation finished. TOP: {TOP}, TON: {TON}, total_loss: {total_loss/batch_cnt :.3f}.')
        FN = [TOP - TP[topk] for topk in range(self.topk-self.min_topk)]
        
        for topk in range(self.topk-self.min_topk):
            if TP[topk] + FN[topk] == 0:
                precision = np.NAN
            else:
                precision = TP[topk] / (TP[topk] + FP[topk])

            if TOP == 0:
                recall = np.NAN
            else:
                recall = TP[topk] / TOP

            F1 = 2 * precision * recall / (precision + recall)
            logger.info(f'[topk={self.min_topk+topk+1}] FP: {FP[topk]}, FN: {FN[topk]}, Precision: {precision: .3f}, Recall: {recall :.3f}, F1-measure: {F1: .3f}.')
    
    def fit(self, train_loader):
        batch_cnt = 0
        total_loss = 0
        model = self.train()
        start = time.time()
        
        for batch in train_loader:
            batch_cnt += 1
            self.optim.zero_grad()
            pred = model.forward(batch)
            label = torch.tensor([next_log['eventid'] for next_log in batch['next']]).to('cuda')
            
            batch_loss = self.loss(pred, label)
            batch_loss.backward()
            total_loss += batch_loss
            self.optim.step()
            
        logger.info(f'Training finished. Train loss: {total_loss/batch_cnt :.3f}, time eplased: {time.time()-start: .3f}s.')
        
    def buildDataLoader(self, 
                        session_train, 
                        session_test, 
                        options):
        test_dataset = LogDataset(session_test, 
                                  options.window_size, 
                                  options.step_size, 
                                  options.num_events)

        logger.info(f'Successfully loaded testing dataset, which has {len(test_dataset)} instances.')

        test_loader = DataLoader(test_dataset, 
                                 collate_fn=collate_fn, 
                                 batch_size=options.eval_batch_size, 
                                 shuffle=False, 
                                 pin_memory=False)
        
        if self.online_mode:
            logger.info('Online mode enabled, apply autoencoder to identify normal instances in testing data.')
            session_normal = {}
            
            for batch in test_loader:
                batch_size = len(batch['session_key'])
                batch_embedding, output = self.autoencoder(batch)
                batch_loss = self.autoencoder_loss(output, batch_embedding).mean(axis=1)

                pred = torch.lt(batch_loss, self.thresh).tolist()

                for ind in range(batch_size):
                    session_key = batch['session_key'][ind]
                    if session_key not in session_normal:
                        session_normal[session_key] = True
                    session_normal[session_key] &= pred[ind]
                    
            normal_sessions = list(filter(lambda key: session_normal[key], session_normal.keys()))
            logger.info(f'{len(normal_sessions)} normal sessions identified by autoencoder.')
            
            for session_key in normal_sessions:
                session_test[session_key]['autoencoder_pred'] = True
                
        train_dataset = LogDataset(session_train, 
                                   options.window_size,
                                   options.step_size, 
                                   options.num_events)

        logger.info(f'Successfully loaded training dataset, which has {len(train_dataset)} instances.')

        train_loader = DataLoader(train_dataset, 
                                  collate_fn=collate_fn,
                                  batch_size=options.batch_size, 
                                  shuffle=False, 
                                  pin_memory=False)
        
        return train_loader, test_loader
        
    def fit_evaluate(self, 
                     session_train, 
                     session_test, 
                     options):
        train_loader, test_loader = self.buildDataLoader(session_train, 
                                                         session_test, 
                                                         options)
        
        for epoch in range(options.n_epoch):
            if getattr(self, 'reset_enabled', False):
                self.resetCandidates()
            
            start = time.time()
            self.fit(train_loader)
            self.evaluate(test_loader)
            logger.info(f'[{epoch+1}|{options.n_epoch}] fit_evaluate finished, time elapsed: {time.time()-start: .3f}s. ')
            
    def setOptimizer(self, optim):
        self.optim = optim
