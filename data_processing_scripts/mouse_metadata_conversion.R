for(i in 1:100){
  metadata = readRDS(paste0("/active/debruinz_project/mouse_data/r_data/mouse_metadata_", i, ".rds"))
  write.csv(metadata, paste0("/active/debruinz_project/mouse_data/python_data/mouse_metadata_", i, ".csv"))
}