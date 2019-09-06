import pathlib
import numpy as np
import pandas as pd

class FacePointDataset:
    def __init__(self, file, transform=None):
        self.transform = transform
        self.images = np.load(file+".npz")["face_images"]
        self.labels = pd.read_csv(file+".csv")

    def __getitem__(self, index):
        if self.transform:
            image = self.transform(self.images[:, :, index])
        else:
            image = self.images[:, :, index]
        label = self.labels.iloc[index].values
        mask_nan = np.isnan(label)
        label[mask_nan] = 0
        return image, label

    def __len__(self):
        return len(self.labels)
