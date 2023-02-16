import io
import itertools
import logging
import re
import torch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def camelCaseSplit(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def split(sentence):
    regex = re.compile('[^a-zA-Z]')
    tokens = regex.sub(' ', sentence).split()
    splitted_tokens = []
    
    for token in tokens:
        splitted_tokens.extend(camelCaseSplit(token))
    
    return splitted_tokens

def buildVocab(parsed_log_df, pretrain_path):
    # Obtain all unique vocabularies from structured log templates
    templates = parsed_log_df['Templates'].tolist()
    template_tokens = [split(template) for template in templates]
    uniq_tokens = set(itertools.chain(*template_tokens))
    logger.info(f'buildVocab identified {len(uniq_tokens)} unique tokens in structured log Dataframe.')
    
    # load pretrained word embeddings
    fin = io.open(pretrain_path, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    word2embedding = {}
    
    for line in fin.readlines():
        tokens = line.rstrip().split()
        word2embedding[tokens[0]] = list(map(float, tokens[1:]))
        
    # Convert template tokens to token_id & build a map from token_id to word embedding
    token2ind = {}
    embeddings = []
    oov_cnt = 0
    
    for token in uniq_tokens:
        if token in word2embedding:
            token2ind[token] = len(token2ind)
            embeddings.append(word2embedding[token])
        else:
            oov_cnt += 1
            
    logger.info(f'{len(token2ind)} tokens can be converted to their corresponding semantic embeddings.') 
    logger.info(f'Number of out-of-vocabulary words is {oov_cnt}.')
    
    # Convert log templates to tokens id and write back to parsed_log_df
    list_tokens_id = []
    
    for tokens in template_tokens:
        tokens_id = []
        
        for token in tokens:
            if token not in token2ind:
                continue
            tokens_id.append(token2ind[token])
            
        list_tokens_id.append(tokens_id)
        
    parsed_log_df['Templates'] = list_tokens_id
    return torch.tensor(embeddings)
