# DESCRIPCIÓN DE LOS DATASETS
# Script con la carga de los datos y la tabla resumen de los mismos
# dodne daremos el num de obs, num variables continuas y categóricas,
# así como el % de cada clase

# ============================================================
# CARGA DE LOS DATOS
# ============================================================
# https://archive.ics.uci.edu/
# install.packages("ucimlrepo")
# install.packages("mlbench")
library(ucimlrepo)
library(mlbench)

# Función para descargar y unir X e y
cargar_uci <- function(id) {
  datos <- fetch_ucirepo(id = id)
  
  X <- datos$data$features
  y <- datos$data$targets
  
  df <- cbind(X, Class = y[[1]])
  df$Class <- as.factor(df$Class)
  
  return(df)
}

# 1) Breast Cancer Wisconsin Diagnostic
datos_wdbc <- cargar_uci(17)

# 2) Wine
datos_wine <- cargar_uci(109)

# 3) Mushroom
datos_mushroom <- cargar_uci(73)

# 4) Waveform 
# 21 variables originales + 19 de ruido
set.seed(123)
n <- 5000
wave <- mlbench.waveform(n)
X_ruido <- matrix(rnorm(n * 19), nrow = n, ncol = 19)
datos_waveform <- data.frame(
  wave$x,
  X_ruido,
  Class = as.factor(wave$classes)
)
colnames(datos_waveform) <- c(paste0("X", 1:40), "Class")

# 5) Statlog Australian Credit
datos_australian <- cargar_uci(143)

# 6) Page Blocks
datos_page <- cargar_uci(78)
# Por seguir el artículo:
datos_page$Class <- ifelse(datos_page$Class == 1, "Negative", "Positive")
datos_page$Class <- as.factor(datos_page$Class)

round(100 * prop.table(table(datos_page$Class)), 2)

# 7) Indian Liver Patient Dataset
datos_IDLP <- cargar_uci(225) 
datos_IDLP <- datos_IDLP[, !names(datos_IDLP) %in% "A/G Ratio", drop = FALSE]

# 8) Online Shoppers Purchasing Intention Dataset
datos_shoppers <- cargar_uci(468)



# ============================================================
# LISTA FINAL
# ============================================================
datasets <- list(
  WDBC = datos_wdbc,
  Wine = datos_wine,
  Mushroom = datos_mushroom,
  Waveform = datos_waveform,
  Australian = datos_australian,
  PageBlocks = datos_page,
  IDLP = datos_IDLP,
  Shoppers = datos_shoppers
  
)
# Dimensión datasets (num variables y num obs)
sapply(datasets, dim)

# Reparto de clases
lapply(datasets, function(df) table(df$Class))

# Reparto de clases en porcentajes
lapply(datasets, function(df) {
  round(100 * prop.table(table(df$Class)), 2)
})



# ============================================================
# TABLA RESUMEN DE LOS DATASETS
# ============================================================
# FUNCIÓN FORMATO REPARTO DE CLASES
formatear_reparto <- function(y) {
  reparto <- round(100 * prop.table(table(y)), 2)
  n_clases <- length(reparto)
  
  # Si son aproximadamente equiprobables
  if (max(reparto) - min(reparto) < 0.05) {
    return(paste0(n_clases, " clases equiprobables (", round(mean(reparto), 2), "%)"))
  }
  
  # Si no, mostramos el reparto detallado
  paste0(names(reparto), ": ", reparto, "%", collapse = "; ")
}


# FUNCIÓN RESUMEN DATASET
resumen_dataset <- function(df, nombre) {
  
  X <- df[, setdiff(names(df), "Class"), drop = FALSE]
  y <- df$Class
  
  data.frame(
    Datos = nombre,
    Observaciones = nrow(df),
    Clases = length(unique(y)),
    Variables = ncol(X),
    Continuas = sum(sapply(X, is.numeric)),
    Categoricas = sum(sapply(X, function(z) is.factor(z) || is.character(z) || is.logical(z))),
    Reparto_clases = formatear_reparto(y),
    row.names = NULL,
    check.names = FALSE
  )
}

# TABLA FINAL
tabla_resumen_final <- do.call(
  rbind,
  Map(resumen_dataset, datasets, names(datasets))
)

# Quitamos nombres de filas explícitamente
rownames(tabla_resumen_final) <- NULL

tabla_resumen_final


# Guardar
write.csv(tabla_resumen_final, "tabla_resumen_datasets_final.csv", row.names = FALSE)

# Ver en R
View(tabla_resumen_final)

# Exportamos en Latex
# install.packages("knitr")
library(knitr)
kable(
  tabla_resumen_final,
  format = "latex",
  booktabs = TRUE,
  caption = "Resumen de los conjuntos de datos considerados."
)
