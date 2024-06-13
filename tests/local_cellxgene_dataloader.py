from sciml.data.local import CellxgeneDataModule
from sciml.utils.constants import REGISTRY_KEYS as RK

def gather_soma_ids(dl):
    soma_ids = set()
    
    num_batches = 0
    for i, batch in enumerate(dl):
        x = batch[0]
        metadata = batch[1]
        
        assert x.shape[0] == 128
        for val in metadata.iloc[: , 0]:
            try:
                assert val not in soma_ids
            except:
                raise RuntimeError("Value", val, "found in soma_ids")
            soma_ids.add(val)
        num_batches = i
    return soma_ids, num_batches


def test_unique_somas_per_split():
    
    datamodule = CellxgeneDataModule(
            num_workers= 2,
            directory_path= '/mnt/projects/debruinz_project/summer_census_data/3m_subset',
            npz_masks= '3m_human*.npz',
            metadata_masks= '3m_human_metadata_*.pkl',
            verbose= True
    )
    
    datamodule.setup(None)
    
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()
    
    train_soma_ids, train_n_batches = gather_soma_ids(train_dl)
    val_soma_ids, val_n_batches = gather_soma_ids(val_dl)
    test_soma_ids, test_n_batches = gather_soma_ids(test_dl)

    assert len(train_soma_ids.intersection(val_soma_ids)) == 0
    train_soma_ids = train_soma_ids.union(val_soma_ids)
    assert len(train_soma_ids.intersection(test_soma_ids)) == 0
    
    print(train_n_batches, val_n_batches, test_n_batches)

test_unique_somas_per_split()