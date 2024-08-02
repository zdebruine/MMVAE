class KLAnnealingFn:
    
    def __init__(self, kl_weight: float):
        self.kl_weight = kl_weight
    
    @property
    def kl_weight(self):
        return self._kl_weight
    
    @kl_weight.setter
    def kl_weight(self, weight):
        self._kl_weight = weight
    
    def step(self) -> None:
        """Override for values that modify the kl_weight per step"""


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
        
        self.m = (max_kl_weight - min_kl_weight) / climax_steps
        self.b = self._min
        
        self.x = -warmup_steps
        
    def step(self) -> None:
        
        self.x += 1
        
        if self.x < 0:
            return
        
        self.kl_weight = min(max(self.m * self.x + self.b, self._min), self._max)