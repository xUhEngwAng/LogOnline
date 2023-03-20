import torch
from embedding import AutoEncoderEmbedding

class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 num_components,
                 num_levels,
                 window_size,
                 thresh):
        super(AutoEncoder, self).__init__()
        self.thresh = thresh
        self.EmbeddingLayer = AutoEncoderEmbedding(num_components, num_levels)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(window_size * (num_components+num_levels+1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 3)
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, window_size * (num_components+num_levels+1)),
            torch.nn.Tanh()
        )
        
    def evaluate(self, dataloader_test):
        TOP = 0
        TON = 0
        TP = 0
        FP = 0

        eval_criterion = torch.nn.MSELoss(reduction='none')
        model = model.eval()

        for batch in dataloader_test:
            batch_size = len(batch['anomaly'])
            batch_embedding, output = model(batch)
            batch_loss = eval_criterion(output, batch_embedding).mean(axis=1)

            pred = torch.lt(batch_loss, self.thresh).tolist()
            is_anomaly = batch['anomaly']
            TOP += batch_size - sum(is_anomaly)

            for ind in range(batch_size):
                if pred[ind]:
                    if is_anomaly[ind]:
                        FP += 1
                    else:
                        TP += 1

            precision = TP / (TP + FP)
            recall = TP / TOP
            F1 = 2 * precision * recall / (precision + recall)

        print(f'Evaluation done, FP: {FP}, precision: {precision: .3f}, recall: {recall :.3f}, F1-measure: {F1: .3f}.')
        
    def forward(self, input_dict):
        embedding_matrix = self.EmbeddingLayer(input_dict)
        embedding = embedding_matrix.view(embedding_matrix.size(0), -1)
        encoding = self.encoder(embedding)
        return embedding, self.decoder(encoding)
    
    def fit(self, dataloader_train):
        batch_cnt = 0
        total_loss = 0
        model = self.train()
        start = time.time()
        
        for batch in dataloader_train:
            batch_cnt += 1

            batch_embedding, output = model(batch)
            batch_loss = criterion(output, batch_embedding)
            # batch_embedding, output, mu, logvar = model(batch)
            # batch_loss = criterion(output, batch_embedding, mu, logvar)

            total_loss += batch_loss.mean()

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
        print(f'Training finished, training loss: {total_loss/batch_cnt :.3f}, time eplased: {time.time() - start: .3f}s.')

class CNNAutoEncoder(torch.nn.Module):
    def __init__(self, 
                 num_components,
                 num_levels,
                 topk):
        # AutoEncoder doesn't support online mode
        super(CNNAutoEncoder, self).__init__()
        self.EmbeddingLayer = AutoEncoderEmbedding(num_components, num_levels)
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3),                               # b, 16, 9, 9
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),                         # b, 16, 5, 5 
            torch.nn.Conv2d(16, 8, 3),                               # b,  8, 3, 3
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=1)                          # b,  8, 2, 2
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 16, 3, stride=2),            # b, 16, 5, 5
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1), # b, 8, 9, 9
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3),                       # b, 1, 11, 11
            torch.nn.Tanh()
        )
        
    def forward(self, input_dict):
        embedding_matrix = self.EmbeddingLayer(input_dict)
        encoding = self.encoder(embedding_matrix)
        return self.decoder(encoding)
    
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, 
                 num_components,
                 num_levels,
                 window_size):
        super(VariationalAutoEncoder, self).__init__()
        self.EmbeddingLayer = AutoEncoderEmbedding(num_components, num_levels)
        self.fc1 = torch.nn.Linear(window_size * (num_components+num_levels+1), 100)
        self.fc21 = torch.nn.Linear(100, 10)
        self.fc22 = torch.nn.Linear(100, 10)
        self.fc3 = torch.nn.Linear(10, 100)
        self.fc4 = torch.nn.Linear(100, window_size * (num_components+num_levels+1))
        
    def encode(self, embedding):
        h1 = torch.nn.functional.relu(self.fc1(embedding))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, input_dict):
        embedding_matrix = self.EmbeddingLayer(input_dict)
        embedding = embedding_matrix.view(embedding_matrix.size(0), -1)
        
        mu, logvar = self.encode(embedding)
        z = self.reparametrize(mu, logvar)
        return embedding, self.decode(z), mu, logvar
    
reconstruction_function = torch.nn.MSELoss(reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD
