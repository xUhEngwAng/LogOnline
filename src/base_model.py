import logging
import numpy as np
import time
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseModel(torch.nn.Module):
    def __init__(self, top_k, online_mode=False):
        super(BaseModel, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = None
        self.top_k = top_k
        self.online_mode = online_mode
    
    def evaluate(self, test_loader):
        if self.online_mode:
            logger.info('Online mode enabled, model would get updated during evaluation.')
            model = self.train()
        else:
            model = self.eval()
            
        session_dict = {}
        
        for batch in test_loader:
            pred = model.forward(batch)
            
            # back-propagation in online mode
            if self.online_mode:
                label = torch.tensor([next_log['eventid'] for next_log in batch['next']]).to('cuda')
                batch_loss = self.loss(pred, label)
                print(f'pred.shape() = {pred.shape}')
                print(f'label.min() = {label.min()}, label.max() = {label.max()}')
                
                self.optim.zero_grad()
                batch_loss.backward()
                #total_loss += batch_loss
                self.optim.step()
            
            _, batch_topk_pred = torch.topk(pred, self.top_k)     
            batch_topk_pred_lst = batch_topk_pred.tolist()
            batch_next_log = [next_log['eventid'] for next_log in batch['next']]
            matched = []
            
            for top_k in range(self.top_k):
                curr_matched = [batch_topk_pred_lst[ind][top_k] == batch_next_log[ind] for ind in range(len(batch_next_log))]
                if 0 < len(matched):
                    prev_matched = matched[-1]
                    curr_matched = [prev|curr for prev, curr in zip(prev_matched, curr_matched)]
                matched.append(curr_matched)
                            
            for ind in range(len(batch['session_key'])):
                session_key = batch['session_key'][ind]
                label = batch['anomaly'][ind]
                if session_key not in session_dict:
                    session_dict[session_key] = {f'matched_{top_k}': True for top_k in range(self.top_k)}
                    session_dict[session_key]['anomaly'] = False
                session_dict[session_key]['anomaly'] |= label
                
                for top_k in range(self.top_k):
                    session_dict[session_key][f'matched_{top_k}'] &= matched[top_k][ind]
        
        TP = [0] * self.top_k
        FP = [0] * self.top_k
        TON = 0 # total negative
        TOP = 0 # total positive
        
        for key, session_info in session_dict.items():
            if session_info['anomaly']:
                TOP += 1
                for top_k in range(self.top_k):
                    if not session_info[f'matched_{top_k}']:
                        TP[top_k] += 1
            else:
                TON += 1
                for top_k in range(self.top_k):
                    if not session_info[f'matched_{top_k}']:
                        FP[top_k] += 1
                    
        logger.info(f'Evaluation finished. TOP: {TOP}, TON: {TON}.')
        FN = [TOP - TP[top_k] for top_k in range(self.top_k)]
        
        for top_k in range(self.top_k):
            if TP[top_k] + FN[top_k] == 0:
                precision = np.NAN
            else:
                precision = TP[top_k] / (TP[top_k] + FP[top_k])

            if TOP == 0:
                recall = np.NAN
            else:
                recall = TP[top_k] / TOP

            F1 = 2 * precision * recall / (precision + recall)
            logger.info(f'[topk={top_k+1}] FP: {FP[top_k]}, FN: {FN[top_k]}, Precision: {precision: .3f}, Recall: {recall :.3f}, F1-measure: {F1: .3f}.')
    
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
        
    def fit_evaluate(self, train_loader, test_loader, n_epoch):
        for epoch in range(n_epoch):
            if getattr(self, 'reset_enabled', False):
                self.resetCandidates()
            
            start = time.time()
            self.fit(train_loader)
            self.evaluate(test_loader)
            logger.info(f'[{epoch+1}|{n_epoch}] fit_evaluate finished, time elapsed: {time.time()-start: .3f}s. ')
            
    def setOptimizer(self, optim):
        self.optim = optim
            