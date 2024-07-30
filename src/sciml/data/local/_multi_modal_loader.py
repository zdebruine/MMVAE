import random
from torch.utils.data import DataLoader
from sciml.utils.constants import REGISTRY_KEYS as RK


class MMDataLoader:

    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders
        
    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        return self
        
                
    def __next__(self):
        # Warn user if next called on when no iterators present
        if not hasattr(self, 'iterators') or not self.iterators:
            import warnings
            warnings.warn("__next__ called on MMDataLoader when iterators are empty")
            raise StopIteration
        
        # get random iterator from iterators available
        iterator = random.choice(self.iterators)
        
        try:
            # try to get a batch from the iterator
            return next(iterator)
        except StopIteration:
            # if iterator is exhausted remove from iterators 
            self.iterators.remove(iterator)
            # if no iterators left through StopIteration
            if not self.iterators:
                raise StopIteration
            return self.__next__()
        