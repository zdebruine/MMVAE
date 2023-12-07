# D-MMVAE

This repository is a base for a research project on diagonal mixture-of-experts variational autoencoding (D-MMVAE). 

### Objective

The objective is to engineer an AI architecture for diagonal integration of multiple modalities by learning a shared latent space that gives cohesive and realistic joint generation across modalities which naturally cannot be encountered jointly.

### Approach

1. Implement a variational autoencoder that generates realistic human single-cell transcriptomes

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/93f54bf3-95b6-4211-822a-62bb72b3849a" width="200">  

2. Add experts (modality-specific encoder/decoders) on either end of the base VAE that can cross-generate

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/48097b88-dfb3-4eec-8d14-0ece9fd50c7f" width="280">  

Test this VAE first on single-cell transcriptomes where transcripts are divided into two batches and presented in a diagonal pattern where ground truth is known, then test on ATAC/RNA where ground truth is partly known, then test on multi-species where discriminators can be trained to assess success.

3. Add adversarial feedback to the encoder to promote cohesive joint generation, and tune adversarial feedback weight to maximize true vs. artificial alignment in cross-generation of ground truth outputs.

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/2434da47-1a17-467d-89a1-614049e2830a" width="300">  

4. Add adversarial feedback to the output of each channel to encourage realistic generated "virtual cells".

<img src="https://github.com/zdebruine/D-MMVAE/assets/2014816/5997b68f-1c53-460c-9764-52cad07c85bf" width="300">  

### Literature Background

D-MMVAE is inspired by works on multi-modal MMVAE from [Shi et. al (2019)](https://arxiv.org/abs/1911.03393), adversarial integration from [Kopp et. al. (2022)](https://www.nature.com/articles/s42256-022-00443-1), recognizing challenges highlighted in [this review](https://www.nature.com/articles/s41467-022-31104-x).
