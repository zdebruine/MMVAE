# from typing import Optional
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal, kl_divergence
# import cmmvae.modules.base as cb


# class CMMVAEModel(nn.Module):
#     def __init__(
#         self,
#         experts: Optional[list[tuple[str, int]]],
#         shared_dim: int,
#         latent_dim: int,
#         batch_keys: Optional[dict[str, int]] = None,
#     ):
#         super().__init__()

#         self.experts = nn.Module(
#             {
#                 expert_name: nn.ModuleDict(
#                     {
#                         "encoder": cb.FCBlock(
#                             cb.FCBlockConfig(
#                                 # TODO: Make feature reduction agnostic of input size
#                                 layers=[expert_input_dim[1], shared_dim, shared_dim],
#                                 use_batch_norm=True,
#                                 activation_fn=nn.ReLU,
#                                 dropout_rate=[0.1, 0.1, 0.0],
#                             )
#                         ),
#                         "decoder": cb.FCBlock(
#                             cb.FCBlockConfig(
#                                 # TODO: Make feature reduction agnostic of input size
#                                 layers=[shared_dim, shared_dim, expert_input_dim[1]],
#                                 use_batch_norm=True,
#                                 activation_fn=nn.ReLU,
#                                 dropout_rate=[0.1, 0.1, 0.0],
#                             )
#                         ),
#                     }
#                 )
#                 for expert_name, expert_input_dim in experts
#             }
#         )

#         shared_layers = [shared_dim, 768, 512, 256]
#         self.shared_encoder = cb.Encoder(
#             latent_dim=latent_dim,
#             return_dist=True,
#             fc_block_config=cb.FCBlockConfig(
#                 # TODO: Make feature reduction agnostic of input size
#                 layers=shared_layers,
#                 use_batch_norm=True,
#                 return_hidden=True,
#                 activation_fn=nn.ReLU,
#             ),
#         )

#         classifiers = OrderedDict()
#         for dc_input_dim in shared_layers + [latent_dim]:
#             layer_classifiers = OrderedDict()
#             for batch_key, n_conditions in batch_keys.items():
#                 layer_classifiers[batch_key] = cb.FCBlock(
#                     cb.FCBlockConfig(
#                         layers=[dc_input_dim, dc_input_dim // 2, n_conditions],
#                         use_layer_norm=True,
#                     )
#                 )
#             classifiers[batch_key] = nn.ModuleDict(layer_classifiers)
#         self.classifiers = nn.ModuleDict(classifiers)

#         self.conditional_layer = cb.ConditionalLayers(
#             fc_block_config=cb.FCBlockConfig(
#                 layers=[latent_dim],
#                 use_layer_norm=True,
#             )
#         )

#         self.shared_decoder = cb.FCBlock(
#             cb.FCBlockConfig(layers=[latent_dim] + shared_layers.reverse())
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         labels: dict[str, torch.Tensor],
#         species: str,
#         kl_weight: float,
#     ):
#         out = self.experts[species]["encoder"](x)
#         qz, z, hidden_representations = self.shared_encoder(out)
#         pz = Normal(torch.zeros_like(z), torch.ones_like(z))
#         z = self.conditional_layer(z)
#         out = self.shared_decoder(z)
#         xhat = self.experts[species]["decoder"](out)

#         batch_size = hidden_representations[0].shape[0]

#         label = torch.empty((batch_size, 2), device=self.device, dtype=torch.float32)

#         label[species == "human", :] = 1

#         adv_losses = []
#         for domain in self.classifiers:
#             label = labels[domain]
#             for i, (hidden_rep, adv) in enumerate(
#                 zip(hidden_representations, self.classifiers)
#             ):
#                 reverse_hidden_rep = cb.GradientReversalFunction.apply(hidden_rep, 1)
#                 adv_output = adv(reverse_hidden_rep)
#                 adv_loss = F.cross_entropy(adv_output, label, reduction="sum")

#         adv_loss = torch.stack(adv_losses).sum()

#         z_kl_div = kl_divergence(qz, pz).sum(dim=-1)

#         if x.layout == torch.sparse_csr:
#             x = x.to_dense()

#         recon_loss = F.mse_loss(xhat, x, reduction="sum")

#         loss = recon_loss + kl_weight * z_kl_div.mean()
