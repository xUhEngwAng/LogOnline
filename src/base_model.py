import logging
import numpy as np
import time
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseModel(torch.nn.Module):
    def __init__(self, topk, online_mode=False):
        super(BaseModel, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = None
        self.topk = topk
        self.bottomk = 10
        self.online_mode = online_mode
    
    def evaluate(self, test_loader):
        if self.online_mode:
            logger.info('Online mode enabled, model would get updated during evaluation.')
            model = self.train()
        else:
            model = self.eval()
            
        session_dict = {}
        batch_cnt = 0
        confident_instances = 0
        total_loss = 0
        
        for batch in test_loader:
            pred = model.forward(batch)
            
            batch_cnt += 1
            batch_size = len(batch['next'])
            
            _, batch_topk_pred = torch.topk(pred, self.topk)
            _, batch_btmk_pred = torch.topk(pred, self.bottomk, largest=False)
            batch_topk_pred_lst = batch_topk_pred.tolist()
            batch_btmk_pred_lst = batch_btmk_pred.tolist()
            batch_next_log = [next_log['eventid'] for next_log in batch['next']]
            
            matched = []
            unmatched = [next_log in btmk_pred for next_log, btmk_pred in zip(batch_next_log, batch_btmk_pred_lst)]
            
            for topk in range(self.topk):
                curr_matched = [batch_topk_pred_lst[ind][topk] == batch_next_log[ind] for ind in range(batch_size)]
                if 0 < len(matched):
                    prev_matched = matched[-1]
                    curr_matched = [prev|curr for prev, curr in zip(prev_matched, curr_matched)]
                matched.append(curr_matched)
                
            # back-propagation in online mode
            '''
            if self.online_mode:
                weight_ = [0 if unmatched[ind] else 1 for ind in range(batch_size)]
                confident_instances += sum(weight_)
                weight = torch.tensor(weight_, dtype=torch.float).to('cuda')
                label = torch.tensor(batch_next_log).to('cuda')
                loss = torch.nn.CrossEntropyLoss(reduction='none')
                batch_loss = loss(pred, label)
                batch_loss = torch.matmul(batch_loss, weight) / (sum(weight_) + 1e-6)
                
                self.optim.zero_grad()
                batch_loss.backward()
                self.optim.step()
            '''
                
            # Use BCELoss as the loss function
            if self.online_mode:
                label = torch.tensor([0 if unmatched[ind] else 1 for ind in range(batch_size)], dtype=torch.float).to('cuda')
                pred_softmax = torch.nn.functional.softmax(pred)
                loss = torch.nn.BCELoss()
                batch_loss = loss(pred_softmax[range(batch_size), batch_next_log], label)
                total_loss += batch_loss
                
                self.optim.zero_grad()
                batch_loss.backward()
                self.optim.step()

            for ind in range(batch_size):
                session_key = batch['session_key'][ind]
                label = batch['anomaly'][ind]
                if session_key not in session_dict:
                    session_dict[session_key] = {f'matched_{topk}': True for topk in range(self.topk)}
                    session_dict[session_key]['anomaly'] = False
                session_dict[session_key]['anomaly'] |= label
                
                for topk in range(self.topk):
                    session_dict[session_key][f'matched_{topk}'] &= matched[topk][ind]
                    
        # logger.info(f'{confident_instances} instances are used for training in online mode.')
        
        TP = [0] * self.topk
        FP = [0] * self.topk
        TON = 0 # total negative
        TOP = 0 # total positive
        
        for key, session_info in session_dict.items():
            if session_info['anomaly']:
                TOP += 1
                for topk in range(self.topk):
                    if not session_info[f'matched_{topk}']:
                        TP[topk] += 1
            else:
                TON += 1
                for topk in range(self.topk):
                    if not session_info[f'matched_{topk}']:
                        FP[topk] += 1
                    
        logger.info(f'Evaluation finished. TOP: {TOP}, TON: {TON}, total_loss: {total_loss/batch_cnt :.3f}.')
        FN = [TOP - TP[topk] for topk in range(self.topk)]
        
        for topk in range(self.topk):
            if TP[topk] + FN[topk] == 0:
                precision = np.NAN
            else:
                precision = TP[topk] / (TP[topk] + FP[topk])

            if TOP == 0:
                recall = np.NAN
            else:
                recall = TP[topk] / TOP

            F1 = 2 * precision * recall / (precision + recall)
            logger.info(f'[topk={topk+1}] FP: {FP[topk]}, FN: {FN[topk]}, Precision: {precision: .3f}, Recall: {recall :.3f}, F1-measure: {F1: .3f}.')
    
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
            
