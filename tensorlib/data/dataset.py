from tensorflow.python.data.ops import dataset_ops
from torchvision.transforms import transforms


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return dataset_ops.ConcatenateDataset()