library(reticulate)
# Ensures that scipy is available
use_virtualenv("/active/debruinz_project/tony_boos/torch_venv/", required = TRUE)
reticulate::py_config()
library(nmslibR)
scipy <- import("scipy")

for (i in 1:100) {

  print(paste0("Chunk ", i))

  dgCMatrix <- readRDS(paste0("/active/debruinz_project/mouse_data/r_data/mouse_chunk_", i, ".rds"))

  # Convert R dgCMatrix to a scipy sparse matrix
  scipy_matrix <- nmslibR::TO_scipy_sparse(dgCMatrix)
  
  # Convert the scipy matrix to CSR format
  csr_matrix <- scipy_matrix$tocsr()
  
  # Save the CSR matrix to an npz file
  scipy$sparse$save_npz(paste0("/active/debruinz_project/mouse_data/python_data/mouse_chunk_", i, ".npz"), csr_matrix)
}
