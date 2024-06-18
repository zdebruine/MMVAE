import torch.nn as nn

class LightningSequential(nn.Sequential):
    def __init__(self, layers: list[nn.Module]):
        print(layers)
        super().__init__(*layers)
        
class LightningLinear(nn.Linear):
    
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)