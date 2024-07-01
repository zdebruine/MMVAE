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
        min_kl_weight: float = 1e-7,
        max_kl_weight: float = 1e-5,
        warmup_steps: float = 1e3,
        climax_steps: float = 1e4,
    ):
        super().__init__(min_kl_weight)
        self._min = min_kl_weight
        self._max = max_kl_weight
        self._climax_steps = climax_steps
        self._warmup_steps = warmup_steps
        self._step = -1
        
    def step(self) -> None:
        self._step += 1
        
        if self.kl_weight >= self._max:
            return
        
        if self._step < self._warmup_steps:
            return
        
        _step = self._step - self._warmup_steps
        
        kl_weight = min(((self._max - self._min) / (self._climax_steps))*(_step) + self._min, self._max)
        
        if kl_weight >= self._max:
            self.kl_weight = self._max
        else:
            self.kl_weight = kl_weight