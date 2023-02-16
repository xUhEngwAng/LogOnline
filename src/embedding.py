import re
import torch

class ContextEmbedding(torch.nn.Module):
    def __init__(self, num_classes, n_dim):
        '''
        Attributes
        ----------
        n_dim: dimension of the output embedding
        '''
        super(ContextEmbedding, self).__init__()
        # self.embedding_table = torch.nn.ParameterDict()
        self.embedding_layer = torch.nn.Embedding(num_classes+1, n_dim)
        self.n_dim = n_dim
        
    def setOptimizer(self, optim):
        '''
        Specify the optimizer used to train the model.
        
        @params optim: the global optimizer used to train the model. EmbeddingLayer dynamically
                       add embeddings during training, which should be registered in `optim` to
                       ensure they are normally updated in back propagation
        '''
        self.optim = optim
        
    def updateEmbeddingSize(self, event_cnt):
        curr_embedding_size = self.embedding_layer.weight.shape[0]
        if event_cnt <= curr_embedding_size:
            return
        
        trgt_embedding_size = curr_embedding_size
        
        while trgt_embedding_size < event_cnt:
            trgt_embedding_size += trgt_embedding_size
            
        new_embeddings = torch.empty(trgt_embedding_size-curr_embedding_size, self.n_dim)
        torch.nn.init.normal_(new_embeddings)
        
        # Concat new_embeddings to self.embedding_layer
        embedding_matrix = torch.concat([self.embedding_layer.weight.data, new_embeddings.cuda()], axis=0)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.optim.param_groups[0]['params'].append(self.embedding_layer.weight.data)
        
    def forward(self, input_dict):
        batch_event_ids = torch.tensor(input_dict['eventids']).cuda()
        return self.embedding_layer(batch_event_ids)
        
    def _forward(self, input_dict):
        batch_event_ids = input_dict['eventids']
        batch_embedding = []
        
        for event_ids in batch_event_ids:
            curr_embedding = []
            
            for event_id in event_ids:
                event_id = str(event_id)
                if event_id not in self.embedding_table:
                    tensor = torch.empty(self.n_dim, requires_grad=True)
                    torch.nn.init.normal_(tensor, 0.1)
                    self.embedding_table[event_id] = torch.nn.Parameter(tensor).cuda()
                    self.optim.param_groups[0]['params'].append(self.embedding_table[event_id])
                    
                curr_embedding.append(self.embedding_table[event_id])
                
            batch_embedding.append(torch.stack(curr_embedding))
                
        return torch.stack(batch_embedding)
    
class OneHotEmbedding(torch.nn.Module):
    def __init__(self, num_classes):
        super(OneHotEmbedding, self).__init__()
        self.num_classes = num_classes
        # reserve a padder embedding for padding index, which is set to `num_classes`
        embedding_matrix = torch.vstack([torch.eye(num_classes), torch.zeros(1, num_classes)])
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
    
    def forward(self, input_dict):
        batch_event_ids = torch.tensor(input_dict['eventids'])
        # convert unseen event ids to self.num_classes
        batch_event_ids[self.num_classes < batch_event_ids] = self.num_classes
        return self.embedding_layer(batch_event_ids.cuda())
    
class SemanticsEmbedding(torch.nn.Module):
    def __init__(self, num_classes, pretrain_matrix, training_tokens_id):
        super(SemanticsEmbedding, self).__init__()
        self.word_embedder = torch.nn.Embedding.from_pretrained(pretrain_matrix, freeze=True).cuda()
        self.num_classes = num_classes
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.nearest_template = {}
        
        # Reserve a zero embedding for template padding
        training_tokens_id.append([])
        template_embeddings = [self.templateEmbedding(tokens_id) for tokens_id in training_tokens_id]
                                   
        self.template_embeddings = torch.vstack(template_embeddings)
        self.template_embedder = torch.nn.Embedding.from_pretrained(self.template_embeddings, freeze=True)
        
    def nearestTemplate(self, event_id, tokens_id):
        '''
        Get the the nearest template's event_id and template embedding for 
        the template denoted by its tokens_id
        '''
        if not event_id in self.nearest_template:
            curr_embedding = self.templateEmbedding(tokens_id)
            cos_sim = self.cos_similarity(self.template_embeddings[:-1], curr_embedding)
            self.nearest_template[event_id] = cos_sim.argmax().item()
        return self.nearest_template.get(event_id)
        
    def templateEmbedding(self, tokens_id):
        '''
        Calculate template embedding by aggregating its tokens' embedding
        '''
        tokens_embedding = self.word_embedder(torch.tensor(tokens_id, dtype=torch.int).cuda())
        if len(tokens_id) == 0:
            return torch.sum(tokens_embedding, axis=0)
        return torch.mean(tokens_embedding, axis=0)
        
    def forward(self, input_dict):
        batch_event_ids = input_dict['eventids']
        batch_tokens_id = input_dict['templates']
        
        # Convert unseen event ids to seen nearest ones
        for batch_ind, event_ids in enumerate(batch_event_ids):
            for ind, event_id in enumerate(event_ids):
                if self.num_classes < event_id:
                    batch_event_ids[batch_ind][ind] = self.nearestTemplate(event_id, batch_tokens_id[batch_ind][ind])
        
        # Apply the same conversion to the next log
        for batch_ind, next_event in enumerate(input_dict['next']):
            if self.num_classes < next_event['eventid']:
                input_dict['next'][batch_ind]['eventid'] = self.nearestTemplate(event_id, next_event['template'])
            
        return self.template_embedder(torch.tensor(batch_event_ids).cuda())
    
class CombinedEmbedding(torch.nn.Module):
    def __init__(self, pretrain_path, freeze=False):
        super(CombinedEmbedding, self).__init__()
        self.embedding_lookup_table = torch.nn.Embedding.from_pretrained(pretrain_path, freeze=freeze)
    
    def forward(self, input_dict):
        pass
        
