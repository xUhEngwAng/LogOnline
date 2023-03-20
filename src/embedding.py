import logging
import re
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AutoEncoderEmbedding(torch.nn.Module):
    def __init__(self, num_components, num_levels):
        super(AutoEncoderEmbedding, self).__init__()
        self.num_components = num_components
        self.num_levels = num_levels
        
        components_embedding = torch.vstack([torch.eye(num_components), torch.zeros(1, num_components)])
        levels_embedding = torch.vstack([torch.eye(num_levels), torch.zeros(1, num_levels)])
        self.component_embedder = torch.nn.Embedding.from_pretrained(components_embedding, freeze=True)
        self.level_embedder = torch.nn.Embedding.from_pretrained(levels_embedding, freeze=True)
        
    def forward(self, input_dict):
        components = torch.tensor(input_dict['components'])
        levels = torch.tensor(input_dict['levels'])
        time_elapsed = torch.tensor(input_dict['time_elapsed']).unsqueeze(-1).cuda()
        
        components[self.num_components < components] = self.num_components
        levels[self.num_levels < levels] = self.num_levels
        
        components_embedding = self.component_embedder(components.cuda())
        levels_embedding = self.level_embedder(levels.cuda())
        return torch.cat([time_elapsed, components_embedding, levels_embedding], dim=2)
                             
class ContextEmbedding(torch.nn.Module):
    def __init__(self, num_classes, n_dim, pretrain_matrix, training_tokens_id):
        
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
                       ensure they are normally updated in back propagation.
        '''
        self.optim = optim
        
    def updateEmbeddingSize(self, new_tokens_id):
        new_embeddings = torch.empty(len(new_tokens_id), self.n_dim)
        torch.nn.init.normal_(new_embeddings)

        # Concat new_embeddings to self.embedding_layer
        embedding_matrix = torch.concat((self.embedding_layer.weight.data, new_embeddings.cuda()), axis=0)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.optim.param_groups[0]['params'].append(self.embedding_layer.weight)
        
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
    def __init__(self, num_classes, n_dim, pretrain_matrix, training_tokens_id):
        super(SemanticsEmbedding, self).__init__()
        self.word_embedder = torch.nn.Embedding.from_pretrained(pretrain_matrix, freeze=True).cuda()
        assert(num_classes == len(training_tokens_id))
        
        # Reserve a zero embedding for template padding
        training_tokens_id.append([])
        template_embeddings = torch.vstack([self.templateEmbedding(tokens_id) for tokens_id in training_tokens_id])
        self.template_embedder = torch.nn.Embedding.from_pretrained(template_embeddings, freeze=True)
        
    def templateEmbedding(self, tokens_id):
        '''
        Calculate template embedding by aggregating its tokens' embedding
        '''
        tokens_embedding = self.word_embedder(torch.tensor(tokens_id, dtype=torch.int).cuda())
        if len(tokens_id) == 0:
            return torch.sum(tokens_embedding, axis=0)
        return torch.mean(tokens_embedding, axis=0)
    
    def forward(self, input_dict):
        batch_event_ids = torch.tensor(input_dict['eventids']).cuda()
        return self.template_embedder(batch_event_ids)
    
    def setOptimizer(self, optim):
        pass
        
    def updateEmbeddingSize(self, new_tokens_id):
        new_template_embeddings = torch.stack([self.templateEmbedding(tokens_id) for tokens_id in new_tokens_id])
        # Concat new_embeddings to self.template_embedder
        embedding_matrix = torch.concat((self.template_embedder.weight.data, new_template_embeddings.cuda()), axis=0)
        self.template_embedder = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
    
class SemanticsNNEmbedding(torch.nn.Module):
    def __init__(self, num_classes, n_dim, pretrain_matrix, training_tokens_id):
        super(SemanticsNNEmbedding, self).__init__()
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
    def __init__(self, num_classes, n_dim, pretrain_matrix, training_tokens_id):
        super(CombinedEmbedding, self).__init__()
        self.context_embedder = ContextEmbedding(num_classes, n_dim, pretrain_matrix, training_tokens_id)  
        self.semantics_embedder = torch.nn.Sequential(
            SemanticsEmbedding(num_classes, n_dim, pretrain_matrix, training_tokens_id),
            torch.nn.Linear(300, n_dim),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(n_dim+n_dim, 1)
    
    def forward(self, input_dict):
        context_embedding = self.context_embedder(input_dict)
        semantics_embedding = self.semantics_embedder(input_dict)
        concat_embedding = torch.concat((context_embedding, semantics_embedding), axis=2)
        alpha = torch.sigmoid(self.fc(concat_embedding))
        return alpha*context_embedding + (1-alpha)*semantics_embedding
    
    def setOptimizer(self, optim):
        self.context_embedder.setOptimizer(optim)
        
    def updateEmbeddingSize(self, new_tokens_id):
        self.context_embedder.updateEmbeddingSize(new_tokens_id)
        self.semantics_embedder[0].updateEmbeddingSize(new_tokens_id)
        
        
