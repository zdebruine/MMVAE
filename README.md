# MMVAE

This repository is a base for a project on Mixture-of-experts Multimodal Variational Autoencoding (MMVAE). 

### Objective

The objective is to engineer an AI architecture for diagonal integration of multiple modalities by learning a shared latent space that gives cohesive and realistic joint generation across modalities which naturally cannot be encountered jointly.

### Architecture

This model is designed for concurrent training on information across diverse modalities.

#### Generative core model

At the core of this model is a variational autoencoder that generates realistic human single-cell transcriptomes:

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/93f54bf3-95b6-4211-822a-62bb72b3849a" width="150">  

####  Expert multi-channel models

For zero-shot cross-generation across multiple modalities (e.g. species), modality-specific encoders and decoders are stacked at either end of the core VAE:

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/48097b88-dfb3-4eec-8d14-0ece9fd50c7f" width="250">  

#### Adversarially-assisted multimodal integration

To help channels communicate with each other and synergize, adversarial feedback will be added to the encoder. This will ensure the core VAE encoder is not biased for one channel or the other:

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/2434da47-1a17-467d-89a1-614049e2830a" width="225">  

#### Adversarial discrimination

Outputs from each channel will also be evaluated with a generative adversarial discriminator network to encourage realistic-looking distributions in context:

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/5997b68f-1c53-460c-9764-52cad07c85bf" width="225">  

### Literature Background

MMVAE is inspired by works on multi-modal MMVAE from [Shi et. al (2019)](https://arxiv.org/abs/1911.03393), adversarial integration from [Kopp et. al. (2022)](https://www.nature.com/articles/s42256-022-00443-1), recognizing challenges highlighted in [this review](https://www.nature.com/articles/s41467-022-31104-x).
