from d_mmvae.data import CellCensusDataLoader
import time

def loader(batch_size, num_workers = 1):
    return CellCensusDataLoader(
        directory_path="/active/debruinz_project/tony_boos/csr_chunks", 
        masks=['chunk_1.npz'], 
        batch_size=batch_size, 
        num_workers=num_workers
    )

def test_batch_size(batch_size):
    assert next(iter(loader(batch_size))).shape[0] == batch_size

# test_batch_size(32)
# print("Batch Shape Test Passed!")
start_time = time.time()
total = 0
for i in loader(32):
    train_input, (expert, kwargs) = i
    print(train_input, expert, kwargs)
    total += 1
print("Entire dataset took", time.time() - start_time, " seconds", total)
    
