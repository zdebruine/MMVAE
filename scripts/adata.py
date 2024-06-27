from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciml.src import sciml
else:
    import sciml
import scanpy as sc        

import cellxgene_census
import anndata

def save_adata_obj():
    census = cellxgene_census.open_soma(census_version="2023-12-15")
    adata = cellxgene_census.get_anndata(
        census=census,
        organism="Homo sapiens",
        obs_value_filter='is_primary_data == True and assay in ["microwell-seq", "10x 3\' v1", "10x 3\' v2", "10x 3\' v3", "10x 3\' transcription profiling", "10x 5\' transcription profiling", "10x 5\' v1", "10x 5\' v2"]',
    )
    sc.write(adata, '/mnt/projects/debruinz_project/integration/adata/unintegrated_data.h5ad')

if __name__ == "__main__":
    
    # vae = BasicVAE()
    # model = VAEModel.load_from_checkpoint('/mnt/projects/debruinz_project/integration/tensorboard/lightning_logs/version_94/checkpoints/epoch=0-val_loss=0.95.ckpt', vae=vae)
    
    adata = sc.read_h5ad('/mnt/projects/debruinz_project/integration/adata/data.h5ad')
    
    latents = model.get_latent_representations(adata, 128)
    
    adata.obsm["X_emb"] = latents
    
    sc.write('/mnt/projects/debruinz_project/integration/adata/integrated_anndata.h5ad', adata)
    
# TODO: FIX FOR MERGED REPO