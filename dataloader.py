import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class LandmarkFeatures(Dataset):
    def __init__(self, filename='train.csv'):
        super(LandmarkFeatures, self).__init__()
        self.data = pd.read_csv(filename)
        self.label_map = {chr(i + 65): i for i in range(26)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        X = torch.tensor(row.iloc[2:].values.astype('float32'))
        y = torch.tensor(self.label_map[row.iloc[1]])
        return X, y
    

if __name__ == '__main__':
    train_dataset = LandmarkFeatures()
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=1)

    X, y = train_dataset.__getitem__(3)
    print(X)
    print(y)