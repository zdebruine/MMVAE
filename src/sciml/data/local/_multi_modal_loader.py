import random
from torch.utils.data import DataLoader
from sciml.utils.constants import REGISTRY_KEYS as RK

class MMLoader:

    def __init__(self, human_dl: DataLoader, mouse_dl: DataLoader):
        self.human_loader = human_dl
        self.mouse_loader = mouse_dl
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        return self.fetch()

    def _get_human_batch(self):
        try:
            batch_dict = next(self.human_iter)
            batch_dict.update({RK.EXPERT_ID: RK.HUMAN})
            return batch_dict
        except StopIteration:
            self.human_exhausted = True
            if self.mouse_exhausted:
                raise StopIteration
            else:
                self.fetch = self._get_mouse_batch
            return self.fetch()

    def _get_mouse_batch(self):
        try:
            batch_dict = next(self.mouse_iter)
            batch_dict.update({RK.EXPERT_ID: RK.MOUSE})
            return batch_dict
        except StopIteration:
            self.mouse_exhausted = True
            if self.human_exhausted:
                raise StopIteration
            else:
                self.fetch = self._get_human_batch
            return self.fetch()
            
    def _get_random_batch(self):
        if random.choice([True, False]):
            return self._get_human_batch()
        else:
            return self._get_mouse_batch()

    def _reset_mouse(self):
        self.mouse_iter = iter(self.mouse_loader)
        self.mouse_exhausted = False

    def _reset_human(self):
        self.human_iter = iter(self.human_loader)
        self.human_exhausted = False

    def reset(self) -> None:
        self._reset_human()
        self._reset_mouse()
        self.fetch = self._get_random_batch

    def shutdown(self):
        self.human_loader.shutdown()
        self.mouse_loader.shutdown()