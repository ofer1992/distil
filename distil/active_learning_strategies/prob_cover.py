import torch

from torch.utils.data import Dataset
from .strategy import Strategy

class ProbCover(Strategy):
    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}) -> None:
        super(ProbCover, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.k = args['k']
        self.delta = args['delta']
        self.update_model(args['embedding'])


    def densest_first(self, unlabeled_embeddings, labeled_embeddings, n):
        unlabeled_embeddings = unlabeled_embeddings.to(self.device)
        labeled_embeddings = labeled_embeddings.to(self.device)
        
        m = unlabeled_embeddings.shape[0]
        # if labeled_embeddings.shape[0] == 0:
            # min_dist = torch.tile(float("inf"), m)
        # else:
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        density = torch.exp(-torch.sort(torch.cdist(unlabeled_embeddings, unlabeled_embeddings, p=2), dim=1)[0][:,:self.k+1].mean(dim=1))
        to_zero = dist_ctr.min(dim=1)[0] < self.delta
        density[to_zero] = 0.
        # import pdb; pdb.set_trace()
        idxs = []
        
        for i in range(n):
            idx = torch.argmax(density)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(unlabeled_embeddings, unlabeled_embeddings[[idx],:])
            # min_dist = torch.minimum(min_dist, dist_new_ctr[:,0])
            to_zero = dist_new_ctr.min(dim=1)[0] < self.delta
            density[to_zero] = 0
                
        return idxs

    def select(self, budget):
        class NoLabelDataset(Dataset):
            
            def __init__(self, wrapped_dataset):
                self.wrapped_dataset = wrapped_dataset
                
            def __getitem__(self, index):
                features, label = self.wrapped_dataset[index]
                return features
            
            def __len__(self):
                return len(self.wrapped_dataset)
        
        self.model.eval()
        embedding_unlabeled = self.get_embedding(self.unlabeled_dataset)
        embedding_labeled = self.get_embedding(NoLabelDataset(self.labeled_dataset))
        chosen = self.densest_first(embedding_unlabeled, embedding_labeled, budget)

        return chosen