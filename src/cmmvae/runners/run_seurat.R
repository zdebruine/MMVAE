# Updated R script to process true and VAE-generated "cis" and "cross" datasets

library(Seurat)
library(dplyr)
library(ggplot2)
library(tibble)
library(ggrepel)

# Define the 5 dataset types and their file patterns
dataset_types <- list(
  true       = list(pattern_h5 = "^full_true.*\\.h5$", pattern_csv = "^full_true.*\\.csv$"),
  cis_nn     = list(pattern_h5 = "^full_cis_nn.*\\.h5$", pattern_csv = "^full_cis_nn.*\\.csv$"),
  cis_dd     = list(pattern_h5 = "^full_cis_dd.*\\.h5$", pattern_csv = "^full_cis_dd.*\\.csv$"),
  cross_nd   = list(pattern_h5 = "^full_cross_nd.*\\.h5$", pattern_csv = "^full_cross_nd.*\\.csv$"),
  cross_dn   = list(pattern_h5 = "^full_cross_dn.*\\.h5$", pattern_csv = "^full_cross_dn.*\\.csv$")
)

# Path to raw data
path <- "/mnt/projects/debruinz_project/disease_study/h5"
output_dir <- "/mnt/projects/debruinz_project/tony_boos/disease_study/expressions"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Storage for DE results
de_list <- list()

# Loop over each dataset type
for (type_name in names(dataset_types)) {
  patterns <- dataset_types[[type_name]]
  h5_files  <- sort(list.files(path, pattern = patterns$pattern_h5, full.names = TRUE))
  csv_files <- sort(list.files(path, pattern = patterns$pattern_csv, full.names = TRUE))

  if (length(h5_files) != length(csv_files)) {
    stop("Mismatch between H5 and CSV files for ", type_name)
  }

  for (i in seq_along(h5_files)) {
    cat("Processing", type_name, "file set", i, "\n")
    data     <- Read10X_h5(h5_files[i])
    metadata <- read.csv(csv_files[i])

    # Create Seurat object and preprocess
    seurat_obj <- CreateSeuratObject(data, meta.data = metadata)
    seurat_obj[["RNA"]]$data <- seurat_obj[["RNA"]]$counts
    seurat_obj$cell_disease <- paste0(seurat_obj$cell_type, " ", seurat_obj$disease)
    Idents(seurat_obj) <- seurat_obj$cell_disease

    # Identify disease and normal groups
    unique_diseases <- setdiff(unique(metadata$disease), "normal")
    if (length(unique_diseases) != 1) {
      message("Skipping file: cannot identify single disease vs normal for ", type_name)
      next
    }
    disease_name <- unique_diseases

    # Loop over cell types for DE
    for (cell in unique(seurat_obj$cell_type)) {
      ident_disease <- paste(cell, disease_name)
      ident_normal  <- paste(cell, "normal")

      if (!(ident_disease %in% Idents(seurat_obj)) || !(ident_normal %in% Idents(seurat_obj))) {
        next
      }

      # Run DE
      de_res <- tryCatch({
        FindMarkers(
          object   = seurat_obj,
          ident.1  = ident_disease,
          ident.2  = ident_normal,
          test.use = "wilcox",
          logfc.threshold = 0.25,
          min.pct  = 0.1
        )
      }, error = function(e) {
        warning("DE error for ", type_name, cell, ": ", e$message)
        NULL
      })

      if (is.null(de_res) || nrow(de_res) == 0) next

      de_df <- de_res %>%
        as.data.frame() %>%
        rownames_to_column("gene") %>%
        rename(log2FC = avg_log2FC) %>%
        mutate(negLog10P = -log10(p_val_adj))

      # Save DE table for later comparison
      de_list[[paste(type_name, cell, sep = "_")]] <- de_df

      # Volcano plot
      p <- ggplot(de_df, aes(x = log2FC, y = negLog10P)) +
        geom_point(aes(color = (abs(log2FC) > 0.25 & p_val_adj < 0.05)), alpha = 0.6, size = 1.5) +
        scale_color_manual(values = c("grey70", "red3"), labels = c("Not Sig.", "Significant"), name = NULL) +
        labs(title = paste(type_name, cell, sep = ": "), x = "Log2 Fold Change", y = "-Log10 Adjusted P") +
        geom_vline(xintercept = c(-0.25, 0.25), linetype = "dashed") +
        geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
        theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5)) +
        geom_text_repel(data = de_df %>% filter(p_val_adj < 0.05) %>% head(10), aes(label = gene), size = 3)

      file_base <- paste(type_name, cell, sep = "_")
      ggsave(file.path(output_dir, paste0(file_base, "_volcano.png")), p, width = 8, height = 6, dpi = 300)

      # Save significant genes table
      sig_out <- de_df %>% filter(abs(log2FC) > 0.25, p_val_adj < 0.05)
      write.csv(sig_out, file.path(output_dir, paste0(file_base, "_sig_genes.csv")), row.names = FALSE)
    }
  }
}

# Compare true vs cis and cross for each cell type
library(VennDiagram)
for (cell in unique(metadata$cell_type)) {
  true_genes <- de_list[[paste("true", cell, sep = "_")]] %>% filter(abs(log2FC) > 0.25, p_val_adj < 0.05) %>% pull(gene)
  cis_nn_genes <- de_list[[paste("cis_nn", cell, sep = "_")]] %>% filter(abs(log2FC) > 0.25, p_val_adj < 0.05) %>% pull(gene)
  cross_nd_genes <- de_list[[paste("cross_nd", cell, sep = "_")]] %>% filter(abs(log2FC) > 0.25, p_val_adj < 0.05) %>% pull(gene)

  venn.plot <- venn.diagram(
    x = list(True = true_genes, Cis = cis_nn_genes, Cross = cross_nd_genes),
    filename = NULL,
    fill = c("red", "blue", "green"),
    alpha = 0.5,
    main = paste("DE overlap for", cell)
  )
  # Save Venn
  png(file.path(output_dir, paste0(cell, "_venn.png")), width = 800, height = 800)
  grid.draw(venn.plot)
  dev.off()
}