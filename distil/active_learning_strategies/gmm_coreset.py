import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from .strategy import Strategy
from .kmeans_sampling import CustomTensorDataset

class GMMCoreset(Strategy):

    """
    Implementation of coreset selection method from GMM Coreset paper.
    
    Parameters
    ----------
    labeled_dataset: torch.utils.data.Dataset
        The labeled training dataset
    unlabeled_dataset: torch.utils.data.Dataset
        The unlabeled pool dataset
    net: torch.nn.Module
        The deep model to use
    nclasses: int
        Number of unique values for the target
    args: dict
        Specify additional parameters
        
        - **batch_size**: The batch size used internally for torch.utils.data.DataLoader objects. (int, optional)
        - **device**: The device to be used for computation. PyTorch constructs are transferred to this device. Usually is one of 'cuda' or 'cpu'. (string, optional)
        - **loss**: The loss function to be used in computations. (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional)
    """    

    def __init__(self, labeled_dataset, unlabeled_dataset, net, nclasses, args={}):
        
        super(GMMCoreset, self).__init__(labeled_dataset, unlabeled_dataset, net, nclasses, args)
        self.k = args['k']
        self.n_kpp = args['n_kpp']
        self.update_model(args['embedding'])


    def get_closest_distances(self, ground_set, center_tensor):
        
        ground_set_loader = DataLoader(ground_set, batch_size = self.args['batch_size'], shuffle = False)

        # Store the minimum distances in this tensor    
        ground_set_min_distances = torch.zeros(len(ground_set)).to(self.device)
        ground_set_closest_center_indices = torch.zeros(len(ground_set), dtype=torch.long).to(self.device)
        evaluated_ground_set_points = 0
    
        with torch.no_grad():
            for ground_set_batch_idx, ground_set_batch in enumerate(ground_set_loader):
                
                # Put the batch on the correct device, calculate embedding, initialize this batch's min distance center to None
                ground_set_batch = CustomTensorDataset(ground_set_batch.to(self.device))
                ground_set_batch = self.get_embedding(ground_set_batch)

                                 
                # Calculate the distance of each point in the ground set batch to each center in the center batch.
                inter_batch_distances = torch.cdist(ground_set_batch, center_tensor, p=2)
                    
                # Calculate the minimum distances across each row; this will reflect the distance to the closest center
                batch_min_distances, batch_min_idx = torch.min(inter_batch_distances, dim=1)                   
                    
                # Assign minimum distance to the correct slice of the storage tensor
                ground_set_min_distances[evaluated_ground_set_points:(evaluated_ground_set_points + len(ground_set_batch))] = batch_min_distances
                ground_set_closest_center_indices[evaluated_ground_set_points:(evaluated_ground_set_points + len(ground_set_batch))] = batch_min_idx
                evaluated_ground_set_points += len(ground_set_batch)

        return ground_set_min_distances, ground_set_closest_center_indices.tolist()

    def kmeans_plusplus(self, num_centers):
        
        # 1. Choose a random point to be the center (uniform dist)
        selected_points = [np.random.choice(len(self.unlabeled_dataset))]
        
        # Keep repeating this step until num_centers centers have been chosen
        while len(selected_points) < num_centers:
            
            # 2. Calculate the squared distance to the nearest center for each point
            selected_centers = Subset(self.unlabeled_dataset, selected_points)
            selected_centers_tensor = self.get_embedding(selected_centers)
            
            ground_set_min_distances, _ = self.get_closest_distances(self.unlabeled_dataset, selected_centers_tensor)
            ground_set_min_distances = torch.pow(ground_set_min_distances, 2)
            
            # 3. Sample a random point with probability proportional to the squared distance
            # Note: torch.multinomial does not require that the weight tensor sum to 1 
            # (e.g., forms a probability distribution). It simply requires non-negative weights 
            # and will form the distribution itself. torch.multinomial can be used as it allows 
            # the tensor to stay on the GPU and because sampling from the multinomial distribution 
            # assigns the probability of sampling element i with the calculated distance weight. 
            distance_probability_distribution = ground_set_min_distances
            random_choice = torch.multinomial(distance_probability_distribution, 1).item()
            
            # 4. Add the chosen index to the center list
            selected_points.append(random_choice)
            
        return selected_points                     

    def gmm_coreset(self, B, alpha, m):
        # dists = torch.cdist(X, B)
        # assignments = torch.argmin(dists, axis=1)
        # closest_dist = dists[np.arange(dists.shape[0]), assignments]**2
        closest_dist, assignments = self.get_closest_distances(self.unlabeled_dataset, B)
        closest_dist = torch.pow(closest_dist, 2)
        assignments = torch.tensor(assignments).to(self.device)
        cluster_dist_sum = torch.tensor([closest_dist[assignments == i].sum() for i in range(B.shape[0])]).to(self.device)
        sizes = torch.unique(assignments, return_counts=True)[1][assignments]
        
        s = alpha*closest_dist + 2*alpha*cluster_dist_sum[assignments]/sizes + 2*closest_dist.sum()/sizes
        s = s / s.sum()
        samples_ind = torch.multinomial(s, m, True)
        # samples_ind = torch.tensor(np.random.choice(np.arange(X.shape[0]), m, True, s))
        return samples_ind, 1/(m*s[samples_ind])

    def select(self, budget):

        """
        Selects next set of points
        
        Parameters
        ----------
        budget: int
            Number of data points to select for labeling
            
        Returns
        ----------
        idxs: list
            List of selected data point indices with respect to unlabeled_dataset
        """	        
        self.model.eval()
        # selecting initial B - centers
        best_inertia = None
        best_centers = None        
        for i in range(self.n_kpp):
            # import pdb; pdb.set_trace()
            centers = self.kmeans_plusplus(self.k)
            centers_subset = Subset(self.unlabeled_dataset, centers)
            centers = self.get_embedding(centers_subset)
            ground_set_min_distances, ground_set_closest_center_indices = self.get_closest_distances(self.unlabeled_dataset, centers)
            inertia = torch.pow(ground_set_min_distances, 2).sum().item()

            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers

        print("after kpp")
        alpha = 16 * np.log2(self.k + 2)
        selected_idx, selected_weights = self.gmm_coreset(best_centers, alpha, budget)

        return selected_idx.tolist()