from torch.utils.data import Dataset


class ConcatDatasets(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        return tuple(dataset[item] for dataset in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
