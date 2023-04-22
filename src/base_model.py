import logging
import numpy as np
import os
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
        self.model_save_path = options.model_save_path
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
            
    def buildDataLoader(self, session_train, session_test, options):
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
            
    def evaluate(self, test_loader):
        if self.online_mode:
            logger.info('Online mode enabled, model would get updated during evaluation.')
            model = self.train()
        else:
            model = self.eval()
            
        min_topk = self.min_topk
        max_topk = self.topk
            
        batch_cnt = 0
        total_loss = 0
        best_result, session_dict = {'F1': 0}, {}
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
                    session_dict[session_key] = {f'matched_{topk}': True for topk in range(min_topk, max_topk)}
                    session_dict[session_key]['anomaly'] = False
                session_dict[session_key]['anomaly'] |= label
                
                for topk in range(min_topk, max_topk):
                    session_dict[session_key][f'matched_{topk}'] &= (batch_ranking_list[ind] <= topk)
                
        for topk in range(min_topk, max_topk):
            result_dict = self._evaluateF1(session_dict, topk, 1)[-1]
            
            if best_result['F1'] < result_dict['F1']:
                best_result = result_dict
                best_result['topk'] = topk
        
        logger.info(f"[topk={best_result['topk']}] TOP: {best_result['TOP']}, TON: {best_result['TON']}, FP: {best_result['FP']}, FN: {best_result['FN']}, Precision: {best_result['precision']: .3f}, Recall: {best_result['recall'] :.3f}, F1-measure: {best_result['F1']: .3f}.")
        
        robustness_result = self._evaluateF1(session_dict, best_result['topk'], 5)
        
        for ind, result_dict in enumerate(robustness_result):
            logger.info(f"[{ind+1}|5] TOP: {result_dict['TOP']}, TON: {result_dict['TON']}, FP: {result_dict['FP']}, FN: {result_dict['FN']}, Precision: {result_dict['precision']: .3f}, Recall: {result_dict['recall'] :.3f}, F1-measure: {result_dict['F1']: .3f}.")
        
        return best_result
    
    def _evaluateF1(self, session_dict, topk, num_slices=1):
        '''
        Evaluate the F1 score of anomaly detection result stored in `session_dict`.
        The sessions are averaged to `num_slices` slices, with the result of each
        slice returned.
        '''
        TP = [0] * num_slices
        FP = [0] * num_slices
        TON = [0] * num_slices  # total negative
        TOP = [0] * num_slices  # total positive
        
        slice_size = int(np.ceil(len(session_dict) / num_slices))
        slices = list(range(0, len(session_dict), slice_size))
        slices.append(len(session_dict))    
        curr_ind = 0
        result = []
        
        for ind, (key, session_info) in enumerate(session_dict.items()):
            if slices[curr_ind + 1] < ind:
                curr_ind += 1
                
            if session_info['anomaly']:
                TOP[curr_ind] += 1
                if not session_info[f'matched_{topk}']:
                    TP[curr_ind] += 1
            else:
                TON[curr_ind] += 1
                if not session_info[f'matched_{topk}']:
                    FP[curr_ind] += 1
                    
        FN = [TOP[slice_ind] - TP[slice_ind] for slice_ind in range(num_slices)]
        
        for slice_ind in range(num_slices):
            if TP[slice_ind] + FP[slice_ind] == 0:
                precision = np.NAN
            else:
                precision = TP[slice_ind] / (TP[slice_ind] + FP[slice_ind])

            if TOP[slice_ind] == 0:
                recall = np.NAN
            else:
                recall = TP[slice_ind] / TOP[slice_ind]

            F1 = 2 * precision * recall / (precision + recall)
            result.append({'TOP': TOP[slice_ind], 'TON': TON[slice_ind], 'FP': FP[slice_ind], 'FN': FN[slice_ind], 'precision': precision, 'recall': recall, 'F1': F1})
            
        return result
    
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
        
    def fit_evaluate(self, 
                     session_train, 
                     session_test, 
                     options):
        train_loader, test_loader = self.buildDataLoader(session_train, session_test, options)
        best_result = {'F1': 0}
        
        for epoch in range(options.n_epoch):
            if getattr(self, 'reset_enabled', False):
                self.resetCandidates()
            
            start = time.time()
            self.fit(train_loader)
            result_dict = self.evaluate(test_loader)
            
            if best_result['F1'] < result_dict['F1']:
                best_result = result_dict
                best_result['epoch'] = epoch+1
                torch.save(self, self.model_save_path)
            
            logger.info(f'[{epoch+1}|{options.n_epoch}] fit_evaluate finished, time elapsed: {time.time()-start: .3f}s.')
            
        logger.info(f"fit_evaluate finished, best result obtained at epoch {best_result['epoch']} with topk = {best_result['topk']}. Precision: {best_result['precision']: .3f}, Recall: {best_result['recall'] :.3f}, F1-measure: {best_result['F1']: .3f}.")
        logger.info(f'Best model saved to {self.model_save_path}.')
            
    def setOptimizer(self, optim):
        self.optim = optim
