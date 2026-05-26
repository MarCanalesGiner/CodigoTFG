# Paquetes de datos
# install.packages(c("rattle", "mlbench", "ucimlrepo"))

library(rattle)
library(mlbench)
library(ucimlrepo)

cargar_uci <- function(id) {
  datos <- fetch_ucirepo(id = id)
  
  X <- datos$data$features
  y <- datos$data$targets
  
  df <- cbind(X, Class = y[[1]])
  df$Class <- as.factor(df$Class)
  
  return(df)
}

# 1) Cargo los datos 'Wine'
data("wine", package = "rattle")
datos_wine <- wine

# 2) Cargo los datos WDBC
datos_wdbc <- cargar_uci(17)

# 3) Cargo Waveform igual que en Datos.R: 21 variables originales + 19 de ruido
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

# 3) Defino una función que me pinte los mapas de calor
library(reshape2)
library(ggplot2)

plot_heatmap <- function(M, titulo) {
  df <- melt(M)
  colnames(df) <- c("Var1", "Var2", "value")
  
  ggplot(df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white", linewidth = 0.15) +
    scale_fill_gradient(
      low = "white",
      high = "#d7191c",
      name = NULL,
      breaks = c(0, 0.25, 0.50, 0.75, 1.00),
      labels = sprintf("%.2f", c(0, 0.25, 0.50, 0.75, 1.00))
    ) +
    coord_equal() +
    labs(title = titulo, x = NULL, y = NULL) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(size = 13, face = "bold", hjust = 0.5, margin = margin(b = 10)),
      axis.text.x = element_text(size = 8, angle = 90, vjust = 0.5, hjust = 1),
      axis.text.y = element_text(size = 8),
      legend.text = element_text(size = 8),
      legend.key.height = grid::unit(0.55, "cm"),
      legend.key.width = grid::unit(0.18, "cm"),
      legend.margin = margin(l = 2, r = 2),
      plot.margin = margin(8, 12, 8, 8),
      panel.grid = element_blank()
    )
}

# Vamos a cargar los paquetes necesarios
# install.packages(c("Hmisc", "energy", "infotheo"))
library(Hmisc)     # Hoeffding
library(energy)    # distance correlation
library(infotheo)  # mutual information

# Seleccionamos solo las variables explicativas
X_wine <- datos_wine[, setdiff(names(datos_wine), "Type")]
X_wdbc <- datos_wdbc[, setdiff(names(datos_wdbc), "Class")]
X_waveform <- datos_waveform[, setdiff(names(datos_waveform), "Class")]


# Funciones para calcular cada matriz de dependencias
# Para cada medida de dependencia queremos una matriz p×p, donde p es el número de variables explicativas.
# M_ij = (d(X_i,X_j)); donde d(X_i,X_j) es la medida de dependencia entre la variable i-ésima y la j-ésima.
mat_pearson <- function(X) {
  cor(X, method = "pearson", use = "pairwise.complete.obs")^2
}

mat_spearman <- function(X) {
  cor(X, method = "spearman", use = "pairwise.complete.obs")^2
}

# Cor solo tiene las correlaciones clásicas, el resto vamos a tener que construir la matriz 'a mano' par a par.
mat_hoeffding <- function(X) {
  p <- ncol(X)
  M <- matrix(0, p, p)
  colnames(M) <- rownames(M) <- colnames(X)
  
  for (i in 1:p) {
    for (j in i:p) {
      if (i == j) {
        M[i, j] <- hoeffd(X[[i]], X[[i]])$D[1,1]
      } else {
        h <- hoeffd(X[[i]], X[[j]])
        val <- h$D[1, 2]
        M[i, j] <- val
        M[j, i] <- val
      }
    }
  }
  M
}

mat_dcor <- function(X) {
  p <- ncol(X)
  M <- matrix(0, p, p)
  colnames(M) <- rownames(M) <- colnames(X)
  
  for (i in 1:p) {
    for (j in i:p) {
      if (i == j) {
        M[i, j] <- dcor(X[[i]], X[[i]])
      } else {
        val <- dcor(X[[i]], X[[j]])
        M[i, j] <- val
        M[j, i] <- val
      }
    }
  }
  M
}


# MI (Información Mutua)
# DUDA MAR: EL PAQUETE TRABAJA CON VARIABLES DSICRETAS, LO QUE ENCAJA MUY BIEN CON LO QUE VENÍA DICIENDO DE QUE DISCRETIZAMSO LAS VARIABLES
library(discretization)

discretizar_mdlp <- function(df, class_var) {
  df_aux <- df
  df_aux[[class_var]] <- as.factor(df_aux[[class_var]])
  
  vars_x <- setdiff(names(df_aux), class_var)
  vars_num <- vars_x[sapply(df_aux[, vars_x, drop = FALSE], is.numeric)]
  vars_cat <- vars_x[sapply(df_aux[, vars_x, drop = FALSE], function(z) {
    is.factor(z) || is.character(z)
  })]
  
  for (v in vars_cat) {
    df_aux[[v]] <- as.factor(df_aux[[v]])
  }
  
  if (length(vars_num) == 0) {
    return(df_aux)
  }
  
  disc <- discretization::mdlp(df_aux[, c(vars_num, class_var), drop = FALSE])
  df_aux[, vars_num] <- disc$Disc.data[, vars_num]
  df_aux[[class_var]] <- as.factor(df[[class_var]])
  
  df_aux
}

# Guardo df discretizados para el MI
datos_wine_mi <- discretizar_mdlp(datos_wine, "Type")
datos_wdbc_mi <- discretizar_mdlp(datos_wdbc, "Class")
datos_waveform_mi <- discretizar_mdlp(datos_waveform, "Class")


# mdlp necesita incluir también la variable de clase


# Sustituimos solo las variables explicativas por sus versiones discretizadas

# Cálculo MI
mat_mi <- function(X) {
  mutinformation(X = X, method = "emp")
}

# Función para calcular cada matriz de dependencias M (tal y como está descrita en el artículo!!!)
# Debemos de calcular la dependencia de cada par de variables por cada clase y tomar el máximo de estos
# como el elemento (i,j) de mi matriz M (representa así el peor escenario)

build_M_max_over_classes <- function(datos, class_var, measure_fun) {
  # Obtengo las clases
  clases <- unique(datos[[class_var]])
  #Obengo los nombres de las variables explicativas
  X_names <- setdiff(names(datos), class_var)
  
  # Calculo la matriz de dependencias por cada clase
  # Filtra las observaciones de la clase y se queda tb solo con las variables explicativas
  M_list <- lapply(clases, function(cl) {
    X_cl <- datos[datos[[class_var]] == cl, X_names, drop = FALSE]
    # Cosntruyo para esas obs (de esa clase) la matriz de dependencia
    measure_fun(X_cl)
  })
  # El resultado será M_list donde M_list[[1]] me da la matriz de dependencias para la clase 1, 
  # M_list[[2]] la de la clase 2,...
  
  # Construyo mi matriz M!
  p <- length(X_names)
  # Inicializo la matriz, como luego voy a coger máximo, pues las inicializo en -Inf
  M_max <- matrix(-Inf, p, p)
  # Pongo los nombres a las filas y a las columnas
  colnames(M_max) <- rownames(M_max) <- X_names
  
  # Guardo el máximo, para ello, uso el comando pmax que me compara dos matrices elemento a elemento
  # y se queda con la matriz cuyas entradas son los máximos
  for (M in M_list) {
    M_max <- pmax(M_max, M, na.rm = TRUE)
  }
  
  M_max[!is.finite(M_max)] <- 0   # por seguridad
  M_max
}


# Cálculo de matrices M para Wine
M_pearson_wine   <- build_M_max_over_classes(datos_wine, "Type",  mat_pearson)
M_spearman_wine  <- build_M_max_over_classes(datos_wine, "Type",  mat_spearman)
M_hoeffding_wine <- build_M_max_over_classes(datos_wine, "Type",  mat_hoeffding)
M_dcor_wine      <- build_M_max_over_classes(datos_wine, "Type",  mat_dcor)
M_mi_wine        <- build_M_max_over_classes(datos_wine_mi, "Type",  mat_mi)


# Cálculo de matrices M para WDBC
M_pearson_wdbc   <- build_M_max_over_classes(datos_wdbc, "Class", mat_pearson)
M_spearman_wdbc  <- build_M_max_over_classes(datos_wdbc, "Class", mat_spearman)
M_hoeffding_wdbc <- build_M_max_over_classes(datos_wdbc, "Class", mat_hoeffding)
M_dcor_wdbc      <- build_M_max_over_classes(datos_wdbc, "Class", mat_dcor)
M_mi_wdbc        <- build_M_max_over_classes(datos_wdbc_mi, "Class", mat_mi)

# Cálculo de matrices M para Waveform
M_pearson_waveform   <- build_M_max_over_classes(datos_waveform, "Class", mat_pearson)
M_spearman_waveform  <- build_M_max_over_classes(datos_waveform, "Class", mat_spearman)
M_hoeffding_waveform <- build_M_max_over_classes(datos_waveform, "Class", mat_hoeffding)
M_dcor_waveform      <- build_M_max_over_classes(datos_waveform, "Class", mat_dcor)
M_mi_waveform        <- build_M_max_over_classes(datos_waveform_mi, "Class", mat_mi)


# Heatmaps de M para Wine, WDBC y Waveform
p1 <- plot_heatmap(M_pearson_wine,   "Wine - Pearson")
p2 <- plot_heatmap(M_spearman_wine,  "Wine - Spearman")
p3 <- plot_heatmap(M_hoeffding_wine, "Wine - Hoeffding")
p4 <- plot_heatmap(M_dcor_wine,      "Wine - dCor")
p5 <- plot_heatmap(M_mi_wine,        "Wine - Informacion mutua")

png("heatmaps_wine.png", width = 3600, height = 2400, res = 300)
library(grid)
library(gridExtra)
grid.arrange(p1, p2, p3, p4, p5, nullGrob(), ncol = 3, padding = grid::unit(1.2, "line"))
dev.off()

j1 <- plot_heatmap(M_pearson_wdbc,   "WDBC - Pearson")
j2 <- plot_heatmap(M_spearman_wdbc,  "WDBC - Spearman")
j3 <- plot_heatmap(M_hoeffding_wdbc, "WDBC - Hoeffding")
j4 <- plot_heatmap(M_dcor_wdbc,      "WDBC - dCor")
j5 <- plot_heatmap(M_mi_wdbc,        "WDBC - Informacion mutua")

png("heatmaps_wdbc.png", width = 3600, height = 2400, res = 300)
grid.arrange(j1, j2, j3, j4, j5, nullGrob(), ncol = 3, padding = grid::unit(1.2, "line"))
dev.off()

k1 <- plot_heatmap(M_pearson_waveform,   "Waveform - Pearson")
k2 <- plot_heatmap(M_spearman_waveform,  "Waveform - Spearman")
k3 <- plot_heatmap(M_hoeffding_waveform, "Waveform - Hoeffding")
k4 <- plot_heatmap(M_dcor_waveform,      "Waveform - dCor")
k5 <- plot_heatmap(M_mi_waveform,        "Waveform - Informacion mutua")

png("heatmaps_waveform.png", width = 3600, height = 2400, res = 300)
grid.arrange(k1, k2, k3, k4, k5, nullGrob(), ncol = 3, padding = grid::unit(1.2, "line"))
dev.off()


# ----------------------------------------------------------------------
# Construcción de la matriz de disimilitud H
build_H_from_M <- function(M) {
  M_star <- max(M, na.rm = TRUE)
  H <- (M_star - M) / M_star
  diag(H) <- 0
  H
}


# Ejemplo de H y dendrograma
H_mi_wine <- build_H_from_M(M_mi_wine)
hc_wine <- hclust(as.dist(H_mi_wine))                 #el método por defecto de hclust es "complete", 
                                                      #lo especifico para dejarlo más claro

graphics.off()
plot(hc_wine, main = "Dendrograma - Wine (MI)")

                                                             


