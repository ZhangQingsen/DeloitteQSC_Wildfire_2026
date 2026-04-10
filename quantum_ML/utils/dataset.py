import torch
from torch.utils.data import Dataset

class WildfireDataset(Dataset):
    """
    PyTorch Dataset for tabular wildfire risk data.
    X: numpy array or pandas DataFrame, shape (N, D)
    y: numpy array or pandas Series, shape (N,)
    """

    def __init__(self, X, y):
        # Convert to torch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Ensure y has shape (N, 1)
        if len(self.y.shape) == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
