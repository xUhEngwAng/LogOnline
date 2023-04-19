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
    def __init__(self, options):
        super(DeepLog, self).__init__(options)
        self.EmbeddingLayer = OneHotEmbedding(options.num_events+1)
        self.FC = torch.nn.Linear(options.hidden_size, options.num_events)
        self.LSTMLayer = torch.nn.LSTM(options.num_events+1, options.hidden_size, options.num_layers, batch_first=True)
        
    def forward(self, input_dict):
        context_embedding = self.EmbeddingLayer(input_dict)
        output, (hn, cn) = self.LSTMLayer(context_embedding)
        return self.FC(hn[-1])
    
class LogAnomaly(BaseModel):
    def __init__(self, options):
        super(LogAnomaly, self).__init__(options)
        self.num_classes = options.num_events
        self.EmbeddingLayer = SemanticsNNEmbedding(options.num_events,
                                                   options.input_size, 
                                                   options.embedding_matrix, 
                                                   options.training_tokens_id)
        self.FC = torch.nn.Linear(options.hidden_size, options.num_events)
        self.LSTMLayer = torch.nn.LSTM(300, options.hidden_size, options.num_layers, batch_first=True)
        
    def forward(self, input_dict):
        context_embedding = self.EmbeddingLayer(input_dict)
        output, (hn, cn) = self.LSTMLayer(context_embedding)
        return self.FC(hn[-1])
    
class UniLog(BaseModel):
    def __init__(self, options):
        super(UniLog, self).__init__(options)
        input_size = options.input_size
        hidden_size = options.hidden_size
        num_classes = options.num_events
        
        if options.embedding_method == 'context':
            self.EmbeddingLayer = ContextEmbedding(num_classes, input_size)
        elif options.embedding_method == 'semantics':
            input_size = 300
            hidden_size = 300
            self.EmbeddingLayer = SemanticsEmbedding(num_classes, options.embedding_matrix, options.training_tokens_id)
        elif options.embedding_method == 'combined':
            self.EmbeddingLayer = CombinedEmbedding(num_classes, 
                                                    input_size, 
                                                    options.embedding_matrix, 
                                                    options.training_tokens_id)
        else:
            logger.error(f'Fatal error, unrecognised embedding method {options.embedding_method} for UniLog model.')
            exit(0)
            
        self.LSTMLayer = torch.nn.LSTM(input_size, hidden_size, options.num_layers, batch_first=True)
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
