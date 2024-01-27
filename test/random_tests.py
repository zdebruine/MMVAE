
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper

for i in DataLoader2(
    datapipe=IterableWrapper(range(0,10)).sharding_filter(),
    reading_service=MultiProcessingReadingService(num_workers=2)
):
    print(i)