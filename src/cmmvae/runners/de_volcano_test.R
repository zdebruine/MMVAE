library(Seurat)
library(dplyr)
library(ggplot2)
library(tibble)
library(ggrepel)

path <- "/mnt/projects/debruinz_project/disease_study/h5"

h5_files <- list.files(path, pattern = "^full.*\\.h5$", full.names = TRUE)
csv_files <- list.files(path, pattern = "^full.*\\.csv$", full.names = TRUE)

h5_files <- sort(h5_files)
csv_files <- sort(csv_files)

for (i in 1:length(h5_files)){
  print(c(h5_files[i],csv_files[i]))

  data <- Read10X_h5(h5_files[i])
  metadata <- read.csv(csv_files[i])

  unique_diseases <- unique(metadata$disease)
  disease_name <- setdiff(unique_diseases, "normal")
  dataset <- unique(metadata$dataset_id)

  print(c(disease_name, dataset))

  seurat_data <- CreateSeuratObject(data, meta.data=metadata)
  seurat_data[["RNA"]]$data <- seurat_data[["RNA"]]$counts

  seurat_data$cell_disease <- paste0(seurat_data$cell_type, " ", seurat_data$disease)
  Idents(seurat_data) <- seurat_data$cell_disease

  cell_types <- unique(seurat_data$cell_type)

  output_dir <- "/mnt/projects/debruinz_project/tony_boos/disease_study/expressions/"
  dir.create(output_dir, showWarnings = FALSE)

  for (cell in cell_types) {
    ident_disease <- paste(cell, disease_name)
    ident_normal  <- paste(cell, "normal")
    
    # Skip if one of the groups doesn't exist
    if (!(ident_disease %in% Idents(seurat_data)) || !(ident_normal %in% Idents(seurat_data))) {
      message("Skipping ", cell, ": one or both groups missing.")
      next
    }
    
    # Run DE
    de_results <- tryCatch({
      FindMarkers(
        object = seurat_data,
        ident.1 = ident_disease,
        ident.2 = ident_normal,
        test.use = "wilcox",
        logfc.threshold = 0.25,
        min.pct = 0.1
      )
    }, error = function(e) {
      message("Error for ", cell, ": ", e$message)
      return(NULL)
    })
    
    # Skip if DE failed or returned NULL
    if (is.null(de_results) || nrow(de_results) == 0) {
      next
    }
    
    fc_cutoff <- 0.25
    p_cutoff  <- 0.05
    
    # Process for plotting
    de_df <- de_results %>%
      as.data.frame() %>%
      rownames_to_column("gene") %>%
      rename(log2FC = avg_log2FC) %>%
      mutate(negLog10P = -log10(p_val_adj))
    
    # Create volcano plot
    p <- ggplot(de_df, aes(x = log2FC, y = negLog10P)) +
      geom_point(aes(
        color = (abs(log2FC) > fc_cutoff & p_val_adj < p_cutoff)
      ),
      alpha = 0.6, size = 1.5
      ) +
      theme_minimal(base_size = 14) +
      scale_color_manual(
        values = c("grey70", "red3"),
        labels = c("Not Sig.", "Significant"),
        name = NULL
      ) +
      labs(
        title = paste("Healthy vs Disease in:", cell),
        x = "Log2 Fold Change",
        y = "-Log10 Adjusted P-Value"
      ) +
      geom_vline(xintercept = c(-fc_cutoff, fc_cutoff), linetype = "dashed") +
      geom_hline(yintercept = -log10(p_cutoff),             linetype = "dashed") +
      geom_text_repel(
        data = de_df %>% filter(p_val_adj < p_cutoff) %>% 
          arrange(p_val_adj) %>% head(10),
        aes(label = gene),
        size = 3
      ) +
      theme(
        plot.title = element_text(hjust = 0.5),
        panel.background = element_rect(fill = "white", color = NA),  # white background
        plot.background  = element_rect(fill = "white", color = NA)
      )
    
    # Save plot
    file_name <- gsub(" ", "_", paste0(disease_name, "_", cell, "_", dataset, "_volcano.png"))
    ggsave(filename = file.path(output_dir, file_name), plot = p, width = 8, height = 6, dpi = 300)

    de_sig <- de_df %>%
      filter(abs(log2FC) > fc_cutoff, p_val_adj < p_cutoff)

    de_sig_out <- de_sig %>%
      select(gene, log2FC, p_val, p_val_adj, negLog10P)

    out_csv <- file.path(
      output_dir,
      paste0(disease_name, "_", cell, "_", dataset, "_significant_genes.csv")
    )

    write.csv(
      de_sig_out,
      file      = out_csv,
      row.names = FALSE
    )
  }
}