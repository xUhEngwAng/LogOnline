import logging
import numpy as np
import time
import torch

from autoencoder import AutoEncoder

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseModel(torch.nn.Module):
    def __init__(self, topk, online_mode=False):
        super(BaseModel, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.optim = None
        self.topk = topk
        self.online_mode = online_mode
        
        if self.online_mode:
            self.thresh = 0.02
            autoencoder = AutoEncoder(9, 6, 11, self.thresh).cuda()
            autoencoder.load_state_dict(torch.load('./checkpoint/bgl_ae_05_newpartition.pth'))
            # autoencoder.load_state_dict(torch.load('./checkpoint/hdfs_ae_08_epoch10.pth'))
            self.autoencoder = autoencoder.eval()
            self.autoencoder_loss = torch.nn.MSELoss(reduction='none')
    
    def evaluate(self, test_loader):
        if self.online_mode:
            logger.info('Online mode enabled, model would get updated during evaluation.')
            model = self.train()
        else:
            model = self.eval()
            
        batch_cnt = 0
        confident_instances = 0
        total_loss = 0
        session_dict = {}
        loss = torch.nn.CrossEntropyLoss(reduction='none')
        
        for batch in test_loader:
            pred = model.forward(batch)
            
            batch_cnt += 1
            batch_size, num_classes = pred.shape[0], pred.shape[1]
            
            batch_next_log = [next_log['eventid'] for next_log in batch['next']]
            batch_ranking = torch.sum(torch.gt(pred.t(), pred[range(batch_size), batch_next_log]), axis=0)
            batch_ranking_list = batch_ranking.tolist()
                
            # back-propagation in online mode
            if self.online_mode:
                batch_embedding, output = self.autoencoder(batch)
                batch_loss = self.autoencoder_loss(output, batch_embedding).mean(axis=1)
                weight = torch.lt(batch_loss, self.thresh).to(torch.float)
                weight_sum = torch.sum(weight).item()
                confident_instances += weight_sum
                   
                label = torch.tensor(batch_next_log).to('cuda')
                batch_loss = loss(pred, label)
                batch_loss = torch.matmul(batch_loss, weight) / (weight_sum + 1e-6)
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
                    session_dict[session_key][f'matched_{topk}'] &= (batch_ranking_list[ind] <= topk)
                    
        logger.info(f'{confident_instances} instances are used for training in online mode.')
        
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
