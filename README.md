# D-MMVAE

This repository is a base for a research project on diagonal mixture-of-experts variational autoencoding (D-MMVAE). 

### Objective

The objective is to engineer an AI architecture for diagonal integration of multiple modalities by learning a shared latent space that gives cohesive and realistic joint generation across modalities which naturally cannot be encountered jointly.

### Approach

1. Implement a variational autoencoder that generates realistic human single-cell transcriptomes
![image](https://github.com/zdebruine/D-MMVAE/assets/2014816/840a4d0b-41cc-499e-b206-f84cffc8c18d)

2. Add experts (modality-specific encoder/decoders) on either end of the base VAE that can cross-generate
![image](https://github.com/zdebruine/D-MMVAE/assets/2014816/0d9ad890-ef2f-43cc-9e2a-3cb1d37c2bcb)

Test this VAE first on single-cell transcriptomes where transcripts are divided into two batches and presented in a diagonal pattern where ground truth is known, then test on ATAC/RNA where ground truth is partly known, then test on multi-species where discriminators can be trained to assess success.

3. Add adversarial feedback to the encoder to promote cohesive joint generation, and tune adversarial feedback weight to maximize true vs. artificial alignment in cross-generation of ground truth outputs.

![image](https://github.com/zdebruine/D-MMVAE/assets/2014816/af74c0a7-5f0c-4144-a660-c1a7d58e23fb)

4. Explore model fine-tuning applications.

### Literature Background

D-MMVAE is inspired by works on multi-modal MMVAE from [Shi et. al (2019)](https://arxiv.org/abs/1911.03393), adversarial integration from [Kopp et. al. (2022)](https://www.nature.com/articles/s42256-022-00443-1), recognizing challenges highlighted in [this review](https://www.nature.com/articles/s41467-022-31104-x).
