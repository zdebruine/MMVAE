class KLAnnealingFn:
    
    def __init__(self, kl_weight: float):
        self.kl_weight = kl_weight
    
    @property
    def kl_weight(self):
        return self._kl_weight
    
    @kl_weight.setter
    def kl_weight(self, weight):
        self._kl_weight = weight


class LinearKLAnnealingFn(KLAnnealingFn):
    
    def __init__(
        self, 
        min_kl_weight: float = 0.1,
        max_kl_weight: float = 1.0,
        climax_step: float = 1.5,
    ):
        super().__init__(min_kl_weight)
        self._min = min_kl_weight
        self._max = max_kl_weight
        self._climax_step = climax_step
        self._step = -1
        
    def step(self) -> None:
        self._step += 1
        
        if self.kl_weight > self._max:
            return
        
        self.kl_weight = min(max((self._max / self._climax_step)*(self._step), self._min), self._max)