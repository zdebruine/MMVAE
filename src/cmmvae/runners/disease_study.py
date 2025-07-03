import scipy.sparse as sp
from cmmvae.runners.cross_generation import CrossGenerator

DISEASES = ["covid", "crohn", "lung_adenocarcinoma"]

model = CrossGenerator("/mnt/projects/debruinz_project/tony_boos/MMVAE/lightning_logs/disease_study/filtered")

for disease in DISEASES:

    data, metadata = model.get_data("/mnt/projects/debruinz_project/disease_study/selected", f"selected_{disease}_disease_study", sample=False)

    out = model.get_cis_outputs(data, metadata, return_z=False)

    sp.save_npz(f"/mnt/projects/debruinz_project/disease_study/selected/{disease}_filtered_out.npz", sp.csr_matrix(out.numpy()))
