import lightning as L

from ._cellxgene_manager import CellxgeneManager

class CellxgeneDataModule(CellxgeneManager, L.LightningDataModule):
    
    def __init__(self, *args, **kwargs):
        super(CellxgeneDataModule, self).__init__(*args, **kwargs)
        super(L.LightningDataModule, self).__init__()
        self.save_hyperparameters(logger=True)