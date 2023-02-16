import logging
import numpy as np
import time
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseModel(torch.nn.Module):
    def __init__(self, top_k):
        super(BaseModel, self).__init__()
        self.top_k = top_k
    
    def evaluate(self, test_loader):
        model = self.eval()
        session_dict = {}
        
        for batch in test_loader:
            pred = model.forward(batch)
            _, batch_topk_pred = torch.topk(pred, self.top_k)
            batch_topk_pred = batch_topk_pred.tolist()
            batch_next_log = [next_log['eventid'] for next_log in batch['next']]
            matched = [next_log in topk_pred for next_log, topk_pred in zip(batch_next_log, batch_topk_pred)]
                
            for ind in range(len(batch['session_key'])):
                session_key = batch['session_key'][ind]
                label = batch['anomaly'][ind]
                if session_key not in session_dict:
                    session_dict[session_key] = {'anomaly': False, 'matched': True}
                session_dict[session_key]['anomaly'] |= label
                session_dict[session_key]['matched'] &= matched[ind]
        
        TP = 0
        FP = 0
        TON = 0 # total negative
        TOP = 0 # total positive
        
        for key, session_info in session_dict.items():
            if session_info['anomaly']:
                TOP += 1
                if not session_info['matched']:
                    TP += 1
            else:
                TON += 1
                if not session_info['matched']:
                    FP += 1
                    
        FN = TOP - TP
        
        if TP + FP == 0:
            precision = np.NAN
        else:
            precision = TP / (TP + FP)
        
        if TOP == 0:
            recall = np.NAN
        else:
            recall = TP / TOP
            
        F1 = 2 * precision * recall / (precision + recall)
        logger.info(f'Evaluation finished. TOP: {TOP}, TON: {TON}, FP: {FP}, FN: {FN}, Precision: {precision: .3f}, Recall: {recall :.3f}, F1-measure: {F1: .3f}.')
    
    def fit(self, train_loader):
        batch_cnt = 0
        total_loss = 0
        model = self.train()
        loss = torch.nn.CrossEntropyLoss()
        start = time.time()
        
        for batch in train_loader:
            batch_cnt += 1
            self.optim.zero_grad()
            pred = model.forward(batch)
            label = torch.tensor([next_log['eventid'] for next_log in batch['next']]).to('cuda')
            
            batch_loss = loss(pred, label)
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
            
    def set_optimizer(self, optim):
        self.optim = optim
            