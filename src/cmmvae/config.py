from typing import Literal, Optional, Union


class GradientClipConfig:
    def __init__(
        self,
        val: Optional[Union[int, float]] = None,
        algorithm: Optional[Union[Literal["norm"], Literal["value"]]] = None,
    ):
        self.val = val
        self.algorithm = algorithm

    def __iter__(self):
        return iter((self.val, self.algorithm))


class AutogradConfig:
    def __init__(
        self,
        adversarial_gradient_clip: Optional[GradientClipConfig] = None,
        vae_gradient_clip: Optional[GradientClipConfig] = None,
        expert_gradient_clip: Optional[GradientClipConfig] = None,
        gan_gradient_clip: Optional[GradientClipConfig] = None,
    ):
        self.adversarial_gradient_clip = adversarial_gradient_clip
        self.vae_gradient_clip = vae_gradient_clip
        self.expert_gradient_clip = expert_gradient_clip
        self.gan_gradient_clip = gan_gradient_clip
