# Mouse data pulled on 2/5/2024

library("tiledb")

library("tiledbsoma")

library("cellxgene.census")

census <- open_soma()

df <- as.data.frame(
  
  census$get("census_data")$get("mus_musculus")$obs$read(
    
    value_filter = "is_primary_data == TRUE",
       
  )$concat())

# filter to only assays mentioning "10x" and including "microwell-seq"

df<- df [grepl("10x", df$assay) | df$assay == "microwell-seq",]

# scramble df

df <- df[sample(1:nrow(df), nrow(df)), ]

# break up df into 100 chunks

n  <- ceiling(nrow(df) / 100)

chunks <- split(df$soma_joinid, rep(1:ceiling(nrow(df)/n), each = n, length.out = nrow(df)))
metadata_chunks <- split(df, rep(1:ceiling(nrow(df)/n), each = n, length.out = nrow(df)))

for(i in 1:length(chunks)){
  
  cat("CHUNK", i, "/", length(chunks),"\n")
  saveRDS(metadata_chunks[[i]], paste0("/active/debruinz_project/mouse_data/r_data/mouse_metadata_", i, ".rds"))
  query <- census$get("census_data")$get("mus_musculus")$axis_query(
    
    measurement_name = "RNA",
    
    obs_query = SOMAAxisQuery$new(coords = chunks[[i]])
    
  )
  
  A <- query$to_sparse_matrix(collection = "X", layer_name = "raw", obs_index = "soma_joinid", var_index = "feature_id" )
  A <- as(A, "dgCMatrix")
  saveRDS(A, paste0("/active/debruinz_project/mouse_data/r_data/mouse_chunk_", i, ".rds"))
  
}
