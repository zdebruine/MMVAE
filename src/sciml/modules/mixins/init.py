import torch.nn as nn


class HeWeightInitMixIn:
    
    _weights_initalized = False
    
    def init_weights(self):
        if self._weights_initalized == True:
            raise RuntimeError("init_weights called twice for one module!")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self._weights_initalized = True