from sciml.cli import SCIMLCli
import scanpy as sc        

from sciml.data import CellxgeneDataManager
import cellxgene_census

if __name__ == "__main__":
    
    from sciml.models import VAE
    
    model = VAE.load_from_checkpoint('/mnt/projects/debruinz_project/integration/tensorboard/lightning_logs/version_45/checkpoints/last.ckpt')
    
    adata = sc.read_h5ad('/mnt/projects/debruinz_project/integration/adata/test_anndata.h5ad')
    
    latents = model.get_latent_representations(adata, 128)
    
    adata.obsm["X_emb"] = latents
    
    sc.write('/mnt/projects/debruinz_project/integration/adata/integrated_anndata.h5ad', adata)