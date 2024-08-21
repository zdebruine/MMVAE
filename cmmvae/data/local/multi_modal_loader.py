import random
from torch.utils.data import DataLoader


class MultiModalDataLoader:
    """
    MultiModalDataLoader allows for iterating over multiple PyTorch DataLoaders simultaneously.
    It randomly selects a batch from one of the provided DataLoaders during each iteration,
    ensuring a diverse mixture of data.

    Attributes:
        dataloaders (tuple[DataLoader]): A collection of PyTorch DataLoaders to iterate over.
    """

    def __init__(self, *dataloaders: DataLoader):
        """
        Initialize the MultiModalDataLoader with a list of DataLoaders.

        Args:
            *dataloaders (DataLoader): A variable number of DataLoader instances to combine.
        """
        self.dataloaders = dataloaders

    def __iter__(self):
        """
        Create an iterator from each DataLoader and return the MultiModalDataLoader instance.

        Returns:
            MultiModalDataLoader: An instance of the MultiModalDataLoader with initialized iterators.
        """
        self.iterators = [iter(dl) for dl in self.dataloaders]
        return self

    def __next__(self):
        """
        Fetch the next batch of data from a randomly selected DataLoader.

        Returns:
            Any: A batch of data from one of the DataLoaders.

        Raises:
            StopIteration: If all DataLoader iterators are exhausted.
            Warning: If __next__ is called without initialized iterators.
        """
        # Warn user if next is called when no iterators are present
        if not hasattr(self, 'iterators') or not self.iterators:
            import warnings
            warnings.warn("'__next__' called on MultiModalDataLoader when iterators are empty")
            raise StopIteration

        # Get a random iterator from the available iterators
        iterator = random.choice(self.iterators)

        try:
            # Try to get a batch from the iterator
            return next(iterator)
        except StopIteration:
            # If the iterator is exhausted, remove it from iterators
            self.iterators.remove(iterator)
            # If no iterators are left, raise StopIteration
            if not self.iterators:
                raise StopIteration
            return self.__next__()
