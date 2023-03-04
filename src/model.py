import logging
import torch

from base_model import BaseModel
from embedding import CombinedEmbedding
from embedding import ContextEmbedding
from embedding import OneHotEmbedding
from embedding import SemanticsEmbedding
from embedding import SemanticsNNEmbedding

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DeepLog(BaseModel):
    def __init__(self, 
                 num_classes, 
                 num_layers, 
                 hidden_size, 
                 top_k,
                 online_mode):
        super(DeepLog, self).__init__(top_k, online_mode)
        self.EmbeddingLayer = OneHotEmbedding(num_classes+1)
        self.FC = torch.nn.Linear(hidden_size, num_classes)
        self.LSTMLayer = torch.nn.LSTM(num_classes+1, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input_dict):
        context_embedding = self.EmbeddingLayer(input_dict)
        output, (hn, cn) = self.LSTMLayer(context_embedding)
        return self.FC(hn[-1])
    
class LogAnomaly(BaseModel):
    def __init__(self, 
                 num_classes,
                 num_layers,
                 input_size,
                 hidden_size,
                 top_k,
                 online_mode,
                 pretrain_matrix, 
                 training_tokens_id):
        super(LogAnomaly, self).__init__(top_k, online_mode)
        self.num_classes = num_classes
        self.EmbeddingLayer = SemanticsNNEmbedding(num_classes, input_size, pretrain_matrix, training_tokens_id)
        self.FC = torch.nn.Linear(hidden_size, num_classes)
        self.LSTMLayer = torch.nn.LSTM(300, hidden_size, num_layers, batch_first=True)
        
    def forward(self, input_dict):
        context_embedding = self.EmbeddingLayer(input_dict)
        output, (hn, cn) = self.LSTMLayer(context_embedding)
        return self.FC(hn[-1])
    
class UniLog(BaseModel):
    def __init__(self, 
                 num_classes,
                 num_layers,
                 input_size, 
                 hidden_size,
                 top_k,
                 online_mode,
                 embedding_method,
                 pretrain_matrix, 
                 training_tokens_id):
        super(UniLog, self).__init__(top_k, online_mode)
        
        if embedding_method == 'context':
            self.EmbeddingLayer = ContextEmbedding(num_classes, input_size, pretrain_matrix, training_tokens_id)
        elif embedding_method == 'semantics':
            input_size = 300
            hidden_size = 300
            self.EmbeddingLayer = SemanticsEmbedding(num_classes, input_size, pretrain_matrix, training_tokens_id)
        elif embedding_method == 'combined':
            self.EmbeddingLayer = CombinedEmbedding(num_classes, input_size, pretrain_matrix, training_tokens_id)
        else:
            logger.error(f'Fatal error, unrecognised embedding method {embedding_method} for UniLog model.')
            exit(0)
            
        self.LSTMLayer = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.num_candidates = 0
        self.num_classes = num_classes
        self.num_embeddings = num_classes + 1
        self.padding_idx = num_classes
        self.reset_enabled = True
        
    def forward(self, input_dict):
        batch_event_ids = input_dict['eventids']
        batch_tokens_id = input_dict['templates']
        new_tokens_id = []
        
        # Update seen candidates by going through each instance in this batch
        for batch_ind, event_ids in enumerate(batch_event_ids):
            for ind, event_id in enumerate(event_ids):
                if self.num_candidates <= event_id and event_id != self.padding_idx:
                    self.num_candidates += 1
                if self.num_embeddings <= event_id:
                    self.num_embeddings += 1
                    new_tokens_id.append(batch_tokens_id[batch_ind][ind])
                    
        for batch_ind, next_event in enumerate(input_dict['next']):
            if self.num_candidates <= next_event['eventid'] and next_event['eventid'] != self.padding_idx:
                self.num_candidates += 1
            if self.num_embeddings <= next_event['eventid']:
                self.num_embeddings += 1
                new_tokens_id.append(next_event['template'])
            if self.padding_idx < next_event['eventid']:
                input_dict['next'][batch_ind]['eventid'] -= 1
        
        if len(new_tokens_id) != 0:
            self.EmbeddingLayer.updateEmbeddingSize(new_tokens_id)
            
        context_embedding = self.EmbeddingLayer(input_dict)
        candidates_embedding = self.EmbeddingLayer({'templates': [[]],
                                                    'eventids': [list(range(self.num_candidates))],
                                                    'next': []})[-1]
        
        output, (hn, cn) = self.LSTMLayer(context_embedding)
        pred = torch.mm(hn[-1], candidates_embedding.t())
        return pred
    
    def resetCandidates(self):
        self.num_candidates = 0
        
    def setOptimizer(self, optim):
        self.EmbeddingLayer.setOptimizer(optim)
        self.optim = optim        

class UniLogNet(BaseModel):
    def __init__(self, 
                 num_layers,
                 input_size, 
                 hidden_size,
                 top_k):
        super(UniLogNet, self).__init__(top_k)
        self.EmbeddingLayer = EmbeddingLayer(input_size)
        self.LSTMLayer = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.key2label = {}
        self.candidates = []
        
    def resetCandidates(self):
        self.candidates.clear()
        self.key2label.clear()
        
    def getEmbeddings(self, log_key_templates):
        '''
        Return a 2D tensor, each row of which represents the embedding of
        the corresponding log_key in log_keys.
        
        @params log_key_templates: a list of {'key': log_key, 'template': log_template}
        '''
        return torch.stack([self.EmbeddingLayer(key_template) for key_template in log_key_templates])
        
    def forward(self, batch_context_logs):
        '''
        batch_context_logs: A list of context_logs
        '''
        labels = []
        batch_candidates_embeddings = []
        num_candidates = []
        
        # For each batch, update the candidates seen by the model by
        # going through the context logs of the first instance.
        for entry in batch_context_logs[0][:-1]:
            key, template = entry.get('key'), entry.get('template')
            self.updateCandidates(key, template)
        
        # For the remaining instances, only the 'next log' is used
        # to update the candidates ( when step_size is set to 1).
        for batch in batch_context_logs:
            key, template = batch[-1].get('key'), batch[-1].get('template')
            self.updateCandidates(key, template)
            labels.append(self.key2label[key])
            batch_candidates_embeddings.append(self.getEmbeddings(self.candidates))
            num_candidates.append(len(self.candidates))

        candidates_embeddings = torch.nn.utils.rnn.pad_sequence(batch_candidates_embeddings, batch_first=True, padding_value=0.0)
        context_embeddings = torch.stack([self.getEmbeddings(context_logs[:-1]) for context_logs in batch_context_logs])
        labels = torch.tensor(labels, dtype=torch.int64).cuda()
        
        output, (hn, cn) = self.LSTMLayer(context_embeddings)
        hn = torch.unsqueeze(hn[-1], dim=2)
        pred = torch.squeeze(torch.bmm(candidates_embeddings, hn), dim=2)
        
        for ind, n_candidate in enumerate(num_candidates):
            pred[ind, n_candidate:] = float('-inf')
                    
        return pred, labels
    
    def updateCandidates(self, key, template):
        if key not in self.key2label:
            self.key2label[key] = len(self.key2label)
            self.candidates.append({'key': key, 'template': template})
        else:
            label = self.key2label[key]
            self.candidates[label] = {'key': key, 'template': template}
