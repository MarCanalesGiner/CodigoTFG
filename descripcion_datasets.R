# 4.1. Descripción de los conjuntos de datos

library(rattle)
library(ucimlrepo)
library(infotheo)
library(discretization)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(dplyr)
library(stringr)

# Creo la carpetas donde se guardan las salidas
dir.create("imagenes", showWarnings = FALSE)
dir.create("tablas", showWarnings = FALSE)

# -----------------------------------------------------------------------
# 1. Cargar datasets disponibles en el repository UCI o directamente en R
# -----------------------------------------------------------------------

cargar_uci <- function(id) {
  datos <- fetch_ucirepo(id = id)
  
  X <- datos$data$features
  y <- datos$data$targets
  
  df <- cbind(X, Class = y[[1]])
  df$Class <- as.factor(df$Class)
  
  return(df)
}

# Breast Cancer Wisconsin Diagnostic (WDBC)
datos_wdbc <- cargar_uci(17)
class_wdbc <- "Class"

# Page Blocks
datos_page <- cargar_uci(78)
# Por seguir el artículo:
datos_page$Class <- ifelse(datos_page$Class == 1, "Negative", "Positive")
datos_page$Class <- as.factor(datos_page$Class)

round(100 * prop.table(table(datos_page$Class)), 2)

class_page <- "Class"

# Mushroom
datos_mushroom <- cargar_uci(73)
class_mushroom <- "Class"

# ------------------------------------------------------------
# 2. Funciones auxiliares
# ------------------------------------------------------------

get_class_split <- function(y) {
  prop <- round(100 * prop.table(table(y)), 2)
  paste(paste0(names(prop), ": ", prop, "%"), collapse = "; ")
}

get_balance_ratio <- function(y) {
  tab <- table(y)
  round(min(tab) / max(tab), 3)
}

count_variable_types <- function(df, class_var) {
  X <- df[, setdiff(names(df), class_var), drop = FALSE]
  
  num_cont <- sum(sapply(X, is.numeric))
  num_cat  <- sum(sapply(X, function(z) is.factor(z) || is.character(z)))
  
  c(L = num_cont, Lprima = num_cat)
}

dataset_summary <- function(df, class_var, dataset_name) {
  tipos <- count_variable_types(df, class_var)
  
  data.frame(
    Dataset = dataset_name,
    Observaciones = nrow(df),
    Clases = length(unique(df[[class_var]])),
    Reparto_clases = get_class_split(df[[class_var]]),
    Balance_min_max = get_balance_ratio(df[[class_var]]),
    Variables_continuas = tipos["L"],
    Variables_categoricas = tipos["Lprima"],
    Total_variables = ncol(df) - 1
  )
}

# ------------------------------------------------------------
# 3. Tabla resumen
# ------------------------------------------------------------

resumen_datos <- bind_rows(
  dataset_summary(datos_wdbc, class_wdbc, "WDBC"),
  dataset_summary(datos_page, class_page, "Page Blocks"),
  dataset_summary(datos_mushroom, class_mushroom, "Mushroom")
)

write.csv(resumen_datos, "tablas/resumen_datasets.csv", row.names = FALSE)

print(resumen_datos)

# Si quieres tabla LaTeX básica:
print(knitr::kable(resumen_datos, format = "latex", booktabs = TRUE))


# ------------------------------------------------------------
# 4. Discretización para información mutua
# ------------------------------------------------------------

discretizar_mdlp <- function(df, class_var) {
  df_aux <- df
  
  # Aseguro que la clase sea factor
  df_aux[[class_var]] <- as.factor(df_aux[[class_var]])
  
  # MDLP solo sobre variables numéricas; las categóricas se dejan como factor/código
  vars_x <- setdiff(names(df_aux), class_var)
  vars_num <- vars_x[sapply(df_aux[, vars_x, drop = FALSE], is.numeric)]
  vars_cat <- vars_x[sapply(df_aux[, vars_x, drop = FALSE], function(z) {
    is.factor(z) || is.character(z)
  })]
  
  # Paso variables factor/character a códigos enteros para que infotheo trabaje bien
  for (v in vars_cat) {
    df_aux[[v]] <- as.factor(df_aux[[v]])
  }
  
  # Aplico MDLP a numéricas si es posible
  # mdlp necesita clase incluida
  if (length(vars_num) == 0) {
    return(df_aux)
  }
  
  out <- discretization::mdlp(df_aux[, c(vars_num, class_var), drop = FALSE])
  df_aux[, vars_num] <- out$Disc.data[, vars_num]
  df_aux[[class_var]] <- as.factor(df[[class_var]])
  
  return(df_aux)
}


# ------------------------------------------------------------
# 5. Matriz M basada en MI: máximo por clases
# ------------------------------------------------------------

mat_mi <- function(X) {
  mutinformation(X = X, method = "emp")
}

build_M_max_over_classes <- function(datos, class_var, measure_fun) {
  clases <- unique(datos[[class_var]])
  X_names <- setdiff(names(datos), class_var)
  
  M_list <- lapply(clases, function(cl) {
    X_cl <- datos[datos[[class_var]] == cl, X_names, drop = FALSE]
    measure_fun(X_cl)
  })
  
  p <- length(X_names)
  M_max <- matrix(-Inf, p, p)
  colnames(M_max) <- rownames(M_max) <- X_names
  
  for (M in M_list) {
    M_max <- pmax(M_max, M, na.rm = TRUE)
  }
  
  M_max[!is.finite(M_max)] <- 0
  
  M_max
}


# ------------------------------------------------------------
# 6. Heatmap
# ------------------------------------------------------------

plot_heatmap_mi <- function(M, titulo) {
  df <- melt(M)
  colnames(df) <- c("Variable_1", "Variable_2", "MI")
  
  ggplot(df, aes(x = Variable_1, y = Variable_2, fill = MI)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "red") +
    coord_equal() +
    labs(title = titulo, x = NULL, y = NULL, fill = "MI") +
    theme_minimal(base_size = 9) +
    theme(
      axis.text.x = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.x = element_blank(),
      axis.ticks.y = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5)
    )
}

generar_heatmap_dataset <- function(df, class_var, dataset_name, file_name) {
  df_disc <- discretizar_mdlp(df, class_var)
  M_mi <- build_M_max_over_classes(df_disc, class_var, mat_mi)
  
  p <- plot_heatmap_mi(
    M_mi,
    paste0(dataset_name)
  )
  
  ggsave(
    filename = paste0("imagenes/", file_name, "_heatmap_MI.png"),
    plot = p,
    width = 7,
    height = 6,
    dpi = 300
  )
  
  return(list(M = M_mi, plot = p))
}


# ------------------------------------------------------------
# 7. Generar heatmaps para datasets cargados
# ------------------------------------------------------------

hm_wdbc <- generar_heatmap_dataset(
  datos_wdbc, class_wdbc,
  "WDBC",
  "wdbc"
)

hm_page <- generar_heatmap_dataset(
  datos_page, class_page,
  "Page Blocks",
  "page blocks"
)

hm_mushroom <- generar_heatmap_dataset(
  datos_mushroom, class_mushroom,
  "Mushroom",
  "mushroom"
)


# -------------------------------
# 8. Figura conjunta para el TFG
# -------------------------------
png("imagenes/heatmaps_MI_resumen.png", width = 2400, height = 1800, res = 300)
grid.arrange(
  hm_wdbc$plot,
  hm_mushroom$plot,
  hm_page$plot,
  ncol = 3
)
dev.off()
