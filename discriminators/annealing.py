

class MonotonicAnnealer():
    def __init__(self, starting_value: float, max: float, rate: float):
        self.start_value = starting_value
        self.rate = rate
        self.max = max
        self.stop = False

    def __call__(self, iteration: int, stop: bool = False):
        if stop or self.stop:
            return 0;
        return min(self.start_value + self.rate * (iteration), self.max)


#https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
#num_iterations: Number of iterations in the training loop.
#num_cycles: Number of cycles to anneal over.
#annealing_fixed_ratio: How much of each epoch will be
class CyclicalAnnealer():
    def __init__(self, minimum: float, maximum: float, num_iterations: int, num_cycles: int,
                 annealing_fixed_ratio: float = 0.5):
        self.minimum = minimum
        self.maximum = maximum

        self.period = int(num_iterations / num_cycles)
        self.annealing_iterations = int(self.period * annealing_fixed_ratio)

        self.step_size = (maximum - minimum) / self.annealing_iterations

        self.stop = False


    def __call__(self, iteration: int, stop: bool = False):
        if stop or self.stop:
            return 0

        relative_iteration = iteration % self.period

        if relative_iteration < self.annealing_iterations:
            return self.minimum + self.step_size * relative_iteration
        else:
            return self.maximum

