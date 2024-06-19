import torch

from sciml.utils.constants import REGISTRY_KEYS as RK

class MMVAEMixIn:
    """
    Defines mmvae forward pass.
    Expectes encoder, decoder, fc_mean, fc_var, and experts to be defined
    """
    
    def cross_generate(self, input_dict):
        cross_gen_dict = {"initial_gen": None,
                          "reversed_gen": None}

        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        expert_id = input_dict[RK.EXPERT]
        other_expert = RK.MOUSE if input_dict[RK.EXPERT] == RK.HUMAN else RK.HUMAN

        x = self.experts[expert_id].encode(x)
        vae_out = self.vae({RK.X: x, RK.METADATA: metadata})
        vae_out[RK.X_HAT] = self.experts[other_expert].decode(vae_out[RK.X_HAT])

        cross_gen_dict["initial_gen"] = vae_out

        x = self.experts[other_expert].encode(vae_out[RK.X_HAT])
        vae_out = self.vae({RK.X: x, RK.METADATA: metadata})
        vae_out[RK.X_HAT] = self.experts[expert_id].decode(vae_out[RK.X_HAT])

        cross_gen_dict["reversed_gen"] = vae_out

        return cross_gen_dict
    
    def forward(self, input_dict):
        
        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        expert_id = input_dict[RK.EXPERT]

        x = self.experts[expert_id].encode(x)
        vae_out = self.vae({RK.X: x, RK.METADATA: metadata})
        vae_out[RK.X_HAT] = self.experts[expert_id].decode(vae_out[RK.X_HAT])

        return vae_out