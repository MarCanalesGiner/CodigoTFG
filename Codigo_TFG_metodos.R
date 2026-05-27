# ============================================================
# Experimentos TFG: Sparse Naive Bayes y comparaciones
# ============================================================
# Objetivo:
#   - Implementar el clasificador Sparse Naive Bayes descrito en el TFG.
#   - Compararlo con Naive Bayes clásico y métodos de selección de variables.
#   - Generar tablas y figuras para el Capítulo 4.
# ============================================================
# Cargamos los datos
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

# BD DESBALANCEADAS
# 6) Page Blocks
datos_page <- cargar_uci(78)
# Por seguir el artículo:
datos_page$Class <- ifelse(datos_page$Class == 1, "Negative", "Positive")
datos_page$Class <- as.factor(datos_page$Class)

round(100 * prop.table(table(datos_page$Class)), 2)

# 7) Indian Liver Patient Dataset (IDLP)
datos_IDLP <- cargar_uci(225) 
datos_IDLP <- datos_IDLP[, !names(datos_IDLP) %in% "A/G Ratio", drop = FALSE]

# 8) Online Shoppers Purchasing Intention Dataset (Shoppers)
datos_shoppers <- cargar_uci(468)


# ------------------------------------------------------------
# 0 ) Cargo los paquetes
# ------------------------------------------------------------

paquetes <- c(
  "caret", "e1071", "klaR", "infotheo", "discretization",
  "dplyr", "tidyr", "purrr", "ggplot2", "reshape2",
  "FSelector", "Boruta", "randomForest", "pROC", "ggrepel"
)

# POR EFICIENCIA: Para paralelizar el lanzamiento para cada uno de los datasets
# library(future.apply)
# plan(multisession, workers = max(1, parallel::detectCores() - 1))

# Instalo las librerías
instalar_si_falta <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
}
invisible(lapply(paquetes, instalar_si_falta))

# Cargo las librerías
invisible(lapply(paquetes, library, character.only = TRUE))
library(ggrepel)

# Carpetas de salida
dir.create("resultados_v2", showWarnings = FALSE)
dir.create("resultados_v2/tablas", showWarnings = FALSE, recursive = TRUE)
dir.create("resultados_v2/figuras", showWarnings = FALSE, recursive = TRUE)

# Fijo semilla para reproducibilidad (Ej: será útil pq por ej  metemos aleatoriedad con CV)
set.seed(123)

# ------------------------------------------------------------
# 1) Funciones auxiliares generales
# ------------------------------------------------------------
# 1) Preparar dataset
# Esta función sirve para que todos los datasets tengan la misma estructura antes de aplicar los métodos:
# Colocamos la variable objetivo en la primera columna, le llamamos Class y la convertimos en tipo factor (así no probelmas
# de que venga como caracteres o algo así)
preparar_dataset <- function(df, class_var = "Class") {
  df <- as.data.frame(df)
  df[[class_var]] <- as.factor(df[[class_var]])
  
  X_names <- setdiff(names(df), class_var)
  df <- df[, c(class_var, X_names), drop = FALSE]
  names(df)[1] <- "Class"
  
  for (v in names(df)) {
    if (is.character(df[[v]])) df[[v]] <- as.factor(df[[v]])
  }
  
  return(df)
}


# 2) Medidas de rendimiento
# 2.1. Tasa global de acierto (Accuracy)
accuracy_score <- function(y_true, y_pred) {
  mean(y_true == y_pred)
}


# Cuando las clases son desbalanceadas puede que el accuracy no sea la mejor medida,
# por eso procedo a definir:
# 2.2. Balanced Accuracy: es la media de los recall_k por clase
balanced_accuracy_score <- function(y_true, y_pred) {
  cm <- table(y_true, y_pred)
  clases <- levels(as.factor(y_true))
  recalls <- sapply(clases, function(cl) {
    # los if son por no pillarnos los dedos, por si se dan casos raros, por ej que no aparezca una clase
    if (!cl %in% rownames(cm)) return(NA_real_)
    den <- sum(cm[cl, ])
    if (den == 0) return(NA_real_)
    if (!cl %in% colnames(cm)) return(0)
    cm[cl, cl] / den
  })
  
  # Finalmente hacemos la media de los recalls_k
  mean(recalls, na.rm = TRUE)
}


# 2.3. AUC
# Para definirlo tenemos que saber si trabajamos con dos clases o más
# para ver si usamos el comando roc o el comando multiclass.roc (compara dos a dos) para obtener la curva ROC
# y posteriormente el área bajo la cruva, es decir, el auc.
# Recordemos que buscamos un auc igual 1, sería lo ideal, cerano e inferior a 0.5 indicaría un mal ajuste.
# Metemos if y tryCatch para evitar que rompa el código con algun caso raro/excepcional.
auc_score <- function(y_true, prob_mat, clase_1 = NULL) {
  y_true <- as.factor(y_true)
  
  # Caso binario
  if (nlevels(y_true) == 2) {
    clase_pos <- clase_1
    if (is.null(clase_pos)) {
      clase_pos <- names(which.min(table(y_true)))   #Fijo la clase minoritaria como la clase positiva (por fijarla vaya)
    }
    
    if (!clase_pos %in% colnames(prob_mat)) return(NA_real_)
    
    clase_neg <- setdiff(levels(y_true), clase_pos)
    
    roc_obj <- tryCatch(
      pROC::roc(
        response = y_true,
        predictor = prob_mat[, clase_pos],
        levels = c(clase_neg, clase_pos),
        quiet = TRUE
      ),
      error = function(e) NULL
    )
    
    if (is.null(roc_obj)) return(NA_real_)
    return(as.numeric(pROC::auc(roc_obj)))
  }
  
  # Caso multiclase
  auc_obj <- tryCatch(
    pROC::multiclass.roc(response = y_true, predictor = prob_mat, quiet = TRUE),
    error = function(e) NULL
  )
  if (is.null(auc_obj)) return(NA_real_)
  as.numeric(auc_obj$auc)
}


# 2.4. Recall clase minoritaria (Lo usaremos para bd desbalanceadas)
# Tomamos como clase 1 la clase minoritaria
recall_clase_score <- function(y_true, y_pred, clase) {
  y_true <- as.factor(y_true)
  y_pred <- factor(y_pred, levels = levels(y_true))
  
  cm <- table(y_true, y_pred)
  
  if (is.null(clase) || !clase %in% rownames(cm)) return(NA_real_)
  
  den <- sum(cm[clase, ])
  if (den == 0) return(NA_real_)
  
  if (!clase %in% colnames(cm)) return(0)
  
  cm[clase, clase] / den
}

recall1_score <- function(y_true, y_pred, clase_1) {
  recall_clase_score(y_true, y_pred, clase_1)
}

recall2_score <- function(y_true, y_pred, clase_1) {
  y_true <- as.factor(y_true)
  if (nlevels(y_true) != 2 || is.null(clase_1)) return(NA_real_)
  
  clase_2 <- setdiff(levels(y_true), clase_1)
  if (length(clase_2) != 1) return(NA_real_)
  
  recall_clase_score(y_true, y_pred, clase_2)
}

recall1_cond_recall2_score <- function(y_true, y_pred, clase_1, min_recall2 = 0.60) {
  r1 <- recall1_score(y_true, y_pred, clase_1)
  r2 <- recall2_score(y_true, y_pred, clase_1)
  
  if (is.na(r1) || is.na(r2)) return(NA_real_)
  if (r2 <= min_recall2) return(NA_real_)   #Si no cumple r2 la condición devuelve NA
  
  r1
}
# 3) Elegir métrica
# Función que simplmente nos va a permitir decidir qué métrica de rendimiento vamos a usar
calcular_metrica <- function(y_true, y_pred, prob_mat = NULL,
                             metric = "accuracy", clase_1 = NULL) {
  if (metric == "accuracy") return(accuracy_score(y_true, y_pred))
  if (metric == "balanced_accuracy") return(balanced_accuracy_score(y_true, y_pred))
  if (metric == "recall1") return(recall1_score(y_true, y_pred, clase_1))
  if (metric == "recall1_cond_recall2_0.60") return(recall1_cond_recall2_score(y_true, y_pred, clase_1, 0.60))
  if (metric == "auc") return(auc_score(y_true, prob_mat, clase_1))
  
  stop("Métrica no reconocida.")
}

# ------------------------------------------------------------
# 2) Preprocesamiento y discretización
# ------------------------------------------------------------
# Como ya sabemos, para información mutua necesitamos variables discretas.
# Siguiendo el artículo de Balnquero et al. 2021, se aplica MDLP.

# Discretizamos el conjunto entero de datos (lo mejoramos porque puede introducir leakage)
# discretizar_dataset_global <- function(df) {
#   # Copio los datos para no tocar los originales
#   df_d <- df
#   
#   # Guardo solo las variables explicativas
#   vars_x <- setdiff(names(df_d), "Class")
#   
#   df_d$Class <- as.factor(df_d$Class)
#   
#   # Variables numéricas --> Sí discretizar con MDLP
#   vars_num <- vars_x[sapply(df_d[, vars_x, drop = FALSE], is.numeric)]
#   
#   # Guardo las variables categóricas: factor o character
#   vars_cat <- vars_x[sapply(df_d[, vars_x, drop = FALSE], function(z) {
#     is.factor(z) || is.character(z)
#   })]
#   
#   # # Las categóricas se codifican como enteros
#   # for (v in vars_cat) {
#   #   df_d[[v]] <- as.integer(as.factor(df_d[[v]]))
#   # }
#   
#   for (v in vars_cat) {
#     df_d[[v]] <- as.factor(df_d[[v]])
#   }
#   
#   # Si no hay variables continuas, no aplicamos MDLP
#   if (length(vars_num) == 0) {
#     return(df_d)
#   }
#   
#   # Discretizamos las vars numéricas
#   aux <- df_d[, c(vars_num, "Class"), drop = FALSE]
#   
#   disc <- discretization::mdlp(aux)
#   
#   # Sustituyo mis variables por las variables  discretizadas
#   df_d[, vars_num] <- disc$Disc.data[, vars_num]
#   df_d$Class <- as.factor(df$Class)
#   
#   return(df_d)
# }
# 
# ajustar_discretizador_mdlp <- function(train, class_var = "Class") {
#   train_d <- as.data.frame(train)
#   train_d[[class_var]] <- as.factor(train_d[[class_var]])
#   
#   vars_x <- setdiff(names(train_d), class_var)
#   vars_num <- vars_x[sapply(train_d[, vars_x, drop = FALSE], is.numeric)]
#   vars_cat <- vars_x[sapply(train_d[, vars_x, drop = FALSE], function(z) {
#     is.factor(z) || is.character(z)
#   })]
#   
#   cortes <- list()
#   if (length(vars_num) > 0) {
#     disc <- discretization::mdlp(train_d[, c(vars_num, class_var), drop = FALSE])
#     cortes <- disc$cutp
#     names(cortes) <- vars_num
#   }
#   
#   list(
#     class_var = class_var,
#     class_levels = levels(train_d[[class_var]]),
#     vars_num = vars_num,
#     vars_cat = vars_cat,
#     cortes = cortes
#   )
# }

aplicar_discretizador_mdlp <- function(df, prep) {
  df_d <- as.data.frame(df)
  class_var <- prep$class_var
  
  df_d[[class_var]] <- factor(df_d[[class_var]], levels = prep$class_levels)
  
  for (v in prep$vars_cat) {
    df_d[[v]] <- as.factor(df_d[[v]])
  }
  
  for (v in prep$vars_num) {
    cuts <- prep$cortes[[v]]
    
    if (length(cuts) == 1 && identical(cuts, "All")) {
      df_d[[v]] <- 1
    } else {
      breaks <- c(-Inf, as.numeric(cuts), Inf)
      df_d[[v]] <- as.numeric(cut(
        df_d[[v]],
        breaks = breaks,
        include.lowest = TRUE,
        right = TRUE
      ))
    }
  }
  df_d
}


# ------------------------------------------------------------
# 3) Matriz de dependencia M y matriz de disimilitud H
# ------------------------------------------------------------

# 1) Cálculo de la Información Mutua (MI)
mat_mi <- function(X) {
  mutinformation(X = X, method = "emp")
}

# 2) Construcción de la matriz M (recupero el código de antes)mut
build_M_max_over_classes <- function(datos, measure_fun) {
  # Obtengo las clases
  clases <- unique(datos[["Class"]])
  #Obengo los nombres de las variables explicativas
  X_names <- setdiff(names(datos), "Class")
  
  # Calculo la matriz de dependencias por cada clase
  # Filtra las observaciones de la clase y se queda tb solo con las variables explicativas
  M_list <- lapply(clases, function(cl) {
    X_cl <- datos[datos[["Class"]] == cl, X_names, drop = FALSE]
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

# 3) Construcción de la matriz de disimilitud H (recupero el código de antes)
build_H_from_M <- function(M) {
  M_star <- max(M, na.rm = TRUE)
  H <- (M_star - M) / M_star
  diag(H) <- 0
  H
}


# ------------------------------------------------------------
# 4) Generación de subconjuntos Sparse NB
# ------------------------------------------------------------
# Creamos una función para generar subconjuntos candidatos de variables.
# Esta recibe el dendograma obtenido con hclust (hc), el nombre de las variables,
# el num máximo de combis evaluadas por cada nivel de corte que se va a porbar, 
# que como sabemos, está fijado en 25, y e l parámetro q se fijaba en función de la dependencia
# observada en M. Por ello defino una función auxiliar para saber q

elegir_q <- function(M) {
  prop_dependencia <- mean(M > 0.1, na.rm = TRUE)
  
  if (prop_dependencia >= 0.20) {
    q <- 0.4
  } else {
    q <- 0.6
  }
  
  return(q)
}

# q <- elegir_q(M)   # Elijo q (0.4 ó 0.6)
generar_subconjuntos_sparse <- function(hc, variables, combination_set = 25, q) {
  
  # Decidimos qué alturas del dendograma vamos a evaluar
  # Obs: hc une variables poco a poco en la salturas correspondientes, las cuales guarda. Cada altura
  # representa un posible corte del dendograma.
  # SI el dendograma tiene menos de 100 alturas, es decir, de 100 cortes posibles, los probamos todos
  # si no, probaremos solo 100 cortes representativos (desde la min a la max altura espaciadas)
  # Obs: claro min{p-1,100} se respeta porque si tegno p < 100 variables, evidentemnte, tendré menos de 100 cortes posibles.
  if (length(hc$height) < 100) {
    alturas <- as.numeric(hc$height)
  } else {  # Rejilla uniforme (como en el código del artículo)
    alturas <- seq(
      hc$height[1],
      hc$height[length(hc$height)],
      length.out = 100
    )
  }
  
  
  subconjuntos <- list()   # Inicializo una lista donde iremos guarando los posibles subcojuntos
  contador <- 1
  
  
  # Para cada altura del dendrograma:
  #   1) Se obtienen grupos de variables dependientes.
  #   2) Se genera un conjunto candidato seleccionando como máximo una variable por grupo.
  #   3) Cada grupo se activa con probabilidad q.
  #   4) Se evalúan varios candidatos en validación.
  
  for (height in alturas) {
    
    H <- ifelse(max(hc$height) >= height, height, max(hc$height)) # aseguro que la altura usada no supere la máxima altura del dendrograma
    
    grupos <- cutree(hc, h = H)  # Corto el dendograma para dicha altura
    
    # Guardos los grupos y vamos uno a uno para selecionar una de cada uno de ellos
    lista_grupos <- list()
    for (i in 1:max(grupos)) {
      lista_grupos[[i]] <- variables[grupos == i]
    }
    
    # ANtes vemos cuántas combis posibles hay para dicha altura (Ej: si tengo 3 conjuntos de 2,3 y 2 eltos resp. habrán 2x3x2 combis)
    # Decisión práctica de Reme: es el probelma intratable? Si pinta mal (o sea grande) no hagas expandgrid pq calcula todas las combis posibles
    # y muuucho coste computacional apra nad porque sabemos que combination_set es máximo 25!
    if (length(hc$height) < 90) {
      # number_of_combinations <- nrow(expand.grid(lista_grupos)) --> Poco eficiente calcular ahí todas las posibles cobis, solo quiero saber el número de combis!
      number_of_combinations <- prod(sapply(lista_grupos, length))
    } else {
      number_of_combinations <- 1000
    }
    
    
    
    # ahora la idea es que para las combis que considere (25 aleatorias, o todas)
    # Generamos un número aleatorio entre 0 y 1 por grupo.
    # Si dicho número es menor que q, el grupo se activa e introduce su representante.
    # Si no, ese grupo no aporta ninguna variable.
    
    # Si tengo demasiadas combis, tengo que elegir 25 al azar (por eso aquí el bucel va hasta 25, pero idea la misma que el otro)
    if (number_of_combinations > combination_set) {
      
      for (i in 1:combination_set) {
        vectorpaux <- runif(length(lista_grupos), min = 0, max = 1)   #genero vector con los numeritos al azar entre 0 y 1
        l2 <- sapply(lista_grupos, function(x) sample(x, 1))   #elegir uno por grupo al azar
        
        vars_sel <- ifelse(vectorpaux < q, l2, NA)
        vars_sel <- vars_sel[!is.na(vars_sel)]    #quita los NA's
        
        if (length(vars_sel) > 0) {
          subconjuntos[[contador]] <- vars_sel
          contador <- contador + 1
        }
      }
      
    } else {   # Si hay menso de 25 pues la considero todas (siguiendo la misma idea)
      
      expandgrid <- expand.grid(lista_grupos)
      
      for (i in 1:number_of_combinations) {
        vectorpaux <- runif(length(lista_grupos), min = 0, max = 1)
        
        vars_sel <- ifelse(vectorpaux < q, as.character(expandgrid[i, ]), NA)
        vars_sel <- vars_sel[!is.na(vars_sel)]
        
        if (length(vars_sel) > 0) {
          subconjuntos[[contador]] <- vars_sel
          contador <- contador + 1
        }
      }
    }
  }
  
  subconjuntos[[contador]] <- variables   # Añadimos también el subconjunto formado por todas las variables
  
  claves <- sapply(subconjuntos, function(s) paste(sort(s), collapse = "|"))
  subconjuntos <- subconjuntos[!duplicated(claves)]
  
  subconjuntos
}

# ------------------------------------------------------------
# 5) Entrenamiento y evaluación de Naive Bayes
# ------------------------------------------------------------
# Lo que vamos a hacer ahora es evaluar un subconjunto (antes creamos los subconjuntos de nuestor método, 
# ahora vamos a ver cómo evaluamos nuestro método en uno de ellos)

# Recibe el conjunto de entrenmaiento, el conjunto de validación, las variables (o sea el subconjunto)
# y la métrica que vamos a usar para evaluarlo

ajustar_y_evaluar_nb <- function(train, eval_data, variables,
                                 metric = "accuracy", clase_1 = NULL) {
  variables <- intersect(variables, setdiff(names(train), "Class"))   # para que no haya nada raro en los nombres de las variables
  
  # Caso vacío o sea el nulo de variables?! -> no devuelva nada
  if (length(variables) == 0) {
    return(list(score = NA_real_, accuracy = NA_real_, balanced_accuracy = NA_real_,
                auc = NA_real_, recall1 = NA_real_, recall2 = NA_real_,
                n_vars = 0, pred = NULL))
  }
  
  train_model <- train[, c("Class", variables), drop = FALSE]
  eval_model  <- eval_data[, c("Class", variables), drop = FALSE]
  
  # Entrenamos el Naive Bayes con ese subconjunto
  modelo <- tryCatch(
    e1071::naiveBayes(Class ~ ., data = train_model),
    error = function(e) {
      e1071::naiveBayes(Class ~ ., data = train_model, laplace = 1)   # si fallara laplace 1 como suavizado (se usa en NB)
    }
  )
  
  if (is.null(modelo)) {
    return(list(score = NA_real_, accuracy = NA_real_, balanced_accuracy = NA_real_,
                auc = NA_real_, recall1 = NA_real_, recall2 = NA_real_,
                n_vars = length(variables), pred = NULL))
  }
  
  # Una vez entrenado el modelo, predicimos para los datos de validación
  pred <- tryCatch(
    predict(modelo, newdata = eval_model[, variables, drop = FALSE], type = "class"),
    error = function(e) NULL
  )
  
  # Y calulamos tb sus probs de pertenencia (nos sirve para el AUC)
  prob <- tryCatch(
    predict(modelo, newdata = eval_model[, variables, drop = FALSE], type = "raw"),
    error = function(e) NULL
  )
  
  # Control de errores
  if (is.null(pred)) {
    return(list(score = NA_real_, accuracy = NA_real_, balanced_accuracy = NA_real_,
                auc = NA_real_, recall1 = NA_real_, recall2 = NA_real_,
                n_vars = length(variables), pred = NULL))
  }
  
  # Cálculo de las métricas en el conjunto de validación
  acc <- accuracy_score(eval_model$Class, pred)
  bacc <- balanced_accuracy_score(eval_model$Class, pred)
  auc <- if (!is.null(prob)) auc_score(eval_model$Class, prob, clase_1) else NA_real_
  recall1 <- recall1_score(eval_model$Class, pred, clase_1)
  recall2 <- recall2_score(eval_model$Class, pred, clase_1)
  
  # Saco la métrica que quieren realmente
  score <- calcular_metrica(eval_model$Class, pred, prob, metric, clase_1)
  
  # Devuélveme en una lista todo de todas formas
  list(score = score, accuracy = acc, balanced_accuracy = bacc,
       auc = auc, recall1 = recall1, recall2 = recall2,
       n_vars = length(variables), pred = pred)
}

evaluar_nb_final <- function(train, valid, test, variables,
                             metric = "accuracy", clase_1 = NULL,
                             final_train = NULL, final_test = NULL) {
  train_eval <- if (is.null(final_train)) dplyr::bind_rows(train, valid) else final_train
  test_eval <- if (is.null(final_test)) test else final_test
  ajustar_y_evaluar_nb(train_eval, test_eval, variables, metric = metric, clase_1 = clase_1)
}


# ------------------------------------------------------------
# 6) Método propuesto: Sparse Naive Bayes
# ------------------------------------------------------------
# Vamos a implementar el método propuesto para un fold entero!
# Devuélveme para ese fold toda la información

fit_sparse_nb_fold <- function(train, valid, test,
                               metric = "accuracy",
                               combination_set = 25,
                               clase_1 = NULL,
                               final_train = NULL,
                               final_test = NULL) {
  
  # Guardamos los datos
  train_d <- train
  valid_d <- valid
  test_d  <- test
  
  variables <- setdiff(names(train_d), "Class")
  
  M <- build_M_max_over_classes(train_d, mat_mi)
  H <- build_H_from_M(M)
  hc <- hclust(as.dist(H), method = "complete")
  
  q <- elegir_q(M)
  
  subconjuntos <- generar_subconjuntos_sparse(
    hc = hc,
    variables = variables,
    combination_set = combination_set,
    q = q
  )
  
  # Ahora recorremos todos los subconjuntos candidatos y les aplicamos la función anterior
  # ajustar_y_evaluar_nb 
  eval_valid <- lapply(subconjuntos, function(vars) {
    res <- ajustar_y_evaluar_nb(train_d, valid_d, vars, metric = metric, clase_1 = clase_1)
    data.frame(   # Guardamos los resultados de ese subconjunto en una fila
      variables = paste(vars, collapse = ";"),
      n_vars = res$n_vars,
      score_valid = res$score,
      accuracy_valid = res$accuracy,
      balanced_accuracy_valid = res$balanced_accuracy,
      auc_valid = res$auc,
      recall1 = res$recall1,
      recall2 = res$recall2,
      stringsAsFactors = FALSE
    )
  })
  
  # Uno todas las filas en una sola tabla
  tabla_valid <- bind_rows(eval_valid)
  tabla_valid <- tabla_valid %>% filter(!is.na(score_valid))  # me cargo los NA's
  
  if (nrow(tabla_valid) == 0) {
    return(NULL)
  }
  
  # Elegimos el mejor subconjunto --> Criterio de desempate: mayor rendimiento y, si empata, menos variables
  mejor <- tabla_valid %>%
    arrange(desc(score_valid), n_vars) %>%
    slice(1)
  
  vars_best <- unlist(strsplit(mejor$variables, ";", fixed = TRUE))   # recuperamos las variables de dicho suconjunto elegido
  
  train_eval <- if (is.null(final_train)) dplyr::bind_rows(train_d, valid_d) else final_train
  test_eval <- if (is.null(final_test)) test_d else final_test
  res_test <- ajustar_y_evaluar_nb(train_eval, test_eval, vars_best, metric = metric, clase_1 = clase_1)   # evaluamos en TEST reentrenando con train+valid
  
  # Devuélveme los resultados :)
  list(
    method = "SparseNB",
    variables = vars_best,
    n_vars = res_test$n_vars,
    accuracy = res_test$accuracy,
    balanced_accuracy = res_test$balanced_accuracy,
    auc = res_test$auc,
    recall1 = res_test$recall1,
    recall2 = res_test$recall2,
    score = res_test$score
    # M = M,
    # H = H,
    # hc = hc,
    # valid_table = tabla_valid
  )
}

# ------------------------------------------------------------
# 7) Métodos de comparación
# ------------------------------------------------------------
# Vamos a preparar los métodos con los que nos vamos a comparar en nuestro TFG (ya explicados anteriormente)
# Vamos a crear una función para cada uno de ellos que me devuelva los resultados para un fold

# NB, CFS y Boruta sí venían en el artículo explicados con qué librerías se implementan, sin embargo, SNB y SNB(MAP) no.
# Nosotros hemos implementado en este trabajo el algoritmo de SNB, para poder compararnos con un método más. 

# ------------------------------------------------------------
# 7.1) Naive Bayes clásico
# ------------------------------------------------------------
# Vaya es NB pero considerando todas las variables, aprovecho mis funciones anteriores

fit_classic_nb_fold <- function(train, valid, test, metric = "accuracy",
                                clase_1 = NULL,
                                final_train = NULL,
                                final_test = NULL) {

  train_d <- if (is.null(final_train)) dplyr::bind_rows(train, valid) else final_train
  
  variables <- setdiff(names(train_d), "Class")
  
  res <- evaluar_nb_final(train, valid, test, variables, metric = metric, clase_1 = clase_1,
                          final_train = final_train, final_test = final_test)
  
  list(
    method = "ClassicNB",
    variables = variables,
    n_vars = res$n_vars,
    accuracy = res$accuracy,
    balanced_accuracy = res$balanced_accuracy,
    auc = res$auc,
    recall1 = res$recall1,
    recall2 = res$recall2,
    score = res$score
  )
}

# ------------------------------------------------------------
# 7.2) CFS + Naive Bayes
# ------------------------------------------------------------

fit_cfs_nb_fold <- function(train, valid, test, metric = "accuracy",
                            clase_1 = NULL,
                            final_train = NULL,
                            final_test = NULL) {
  
  train_d <- train
  
  # Usamos la función cfs de la librería FSelector (tal y como nos indica el artículo)
  vars_sel <- tryCatch(
    FSelector::cfs(Class ~ ., data = train_d),
    error = function(e) character(0)
  )
  
  # Si CFS no devuelve variables, usamos todas como fallback.
  if (length(vars_sel) == 0) {
    vars_sel <- setdiff(names(train_d), "Class") 
  }
  
  # Elegido el conjunto de variables con CFS, ajusto NB con ellas
  res <- evaluar_nb_final(train, valid, test, vars_sel, metric = metric, clase_1 = clase_1,
                          final_train = final_train, final_test = final_test)
  
  list(
    method = "CFS",
    variables = vars_sel,
    n_vars = res$n_vars,
    accuracy = res$accuracy,
    balanced_accuracy = res$balanced_accuracy,
    auc = res$auc,
    recall1 = res$recall1,
    recall2 = res$recall2,
    score = res$score
  )
}


# ------------------------------------------------------------
# 7.3) Boruta adaptado a Naive Bayes
# ------------------------------------------------------------
# Como dijimos Boruta fue diseñado para Random Forest, no obstante, puede ser adaptado para cualquier clasificador
# Para adaptarlo usamos la función filterVarImp del paquete caret que nos devuelve la importacia de  cada variable
# Y luego usamos el comando Boruta de Boruta

getImpNB <- function(x, y, ...) {
  
  imp <- caret::filterVarImp(x = x, y = y)
  
  if (is.data.frame(imp) || is.matrix(imp)) {
    imp <- apply(as.matrix(imp), 1, max, na.rm = TRUE)   #nos quedamos con el máximo pq filterVarImp
                                                         #devuelve la importancia de la variable para cada clase
                                                         #si para una clase es relevante, pues, probablemente nos interesará,
                                                         #de ahí lo de coger el máximo
  }
  
  imp[is.na(imp)] <- 0
  return(imp)
}

fit_boruta_nb_fold <- function(train, valid, test, metric = "accuracy", maxRuns = 50,
                               clase_1 = NULL,
                               final_train = NULL,
                               final_test = NULL) {
  
  train_d <- train
  
  bor <- tryCatch(
    Boruta::Boruta(
      Class ~ .,
      data = train_d,
      getImp = getImpNB,
      maxRuns = maxRuns,
      doTrace = 0
    ),
    error = function(e) NULL
  )
  
  if (is.null(bor)) {
    vars_sel <- setdiff(names(train_d), "Class")
  } else {
    vars_sel <- Boruta::getSelectedAttributes(bor, withTentative = TRUE)
    
    if (length(vars_sel) == 0) {
      vars_sel <- setdiff(names(train_d), "Class")
    }
  }
  
  res <- evaluar_nb_final(train, valid, test, vars_sel, metric = metric, clase_1 = clase_1,
                          final_train = final_train, final_test = final_test)
  
  list(
    method = "Boruta",
    variables = vars_sel,
    n_vars = res$n_vars,
    accuracy = res$accuracy,
    balanced_accuracy = res$balanced_accuracy,
    auc = res$auc,
    recall1 = res$recall1,
    recall2 = res$recall2,
    score = res$score
  )
}


# ------------------------------------------------------------
# 7.4) Selective Naive Bayes clásico (SNB) 
# ------------------------------------------------------------
# Implementación wrapper greedy:
# Básicamente aplicamos el algoritmo greedy descrito en el TFG

fit_snb_fold <- function(train, valid, test, metric = "accuracy",
                         clase_1 = NULL,
                         final_train = NULL,
                         final_test = NULL) {

  train_d <- train
  valid_d <- valid
  test_d  <- test
  
  todas_vars <- setdiff(names(train_d), "Class")
  vars_sel <- character(0)
  vars_restantes <- todas_vars
  
  mejor_score <- -Inf
  mejora <- TRUE
  
  while (mejora && length(vars_restantes) > 0) {
    
    candidatos <- lapply(vars_restantes, function(v) {
      vars_try <- c(vars_sel, v)
      res <- ajustar_y_evaluar_nb(
        train_d,
        valid_d,
        vars_try,
        metric = metric,
        clase_1 = clase_1
      )
      
      data.frame(
        variable = v,
        score = res$score,
        n_vars = length(vars_try),
        stringsAsFactors = FALSE
      )
    })
    
    tabla_candidatos <- dplyr::bind_rows(candidatos)
    tabla_candidatos <- tabla_candidatos %>% dplyr::filter(!is.na(score))
    
    if (nrow(tabla_candidatos) == 0) break
    
    mejor_score_candidato <- max(tabla_candidatos$score, na.rm = TRUE)
    mejores_candidatos <- tabla_candidatos %>%
      dplyr::filter(score == mejor_score_candidato)
    mejor_candidato <- mejores_candidatos[sample(seq_len(nrow(mejores_candidatos)), 1), ]
    
    if (mejor_candidato$score >= mejor_score) {
      vars_sel <- c(vars_sel, mejor_candidato$variable)
      vars_restantes <- setdiff(vars_restantes, mejor_candidato$variable)
      mejor_score <- mejor_candidato$score
    } else {
      mejora <- FALSE
    }
  }
  
  if (length(vars_sel) == 0) {
    vars_sel <- todas_vars
  }
  
  train_eval <- if (is.null(final_train)) dplyr::bind_rows(train_d, valid_d) else final_train
  test_eval <- if (is.null(final_test)) test_d else final_test
  
  res_test <- evaluar_nb_final(train_d, valid_d, test_d, vars_sel, metric = metric, clase_1 = clase_1,
                               final_train = final_train, final_test = final_test)
  
  list(
    method = "SNB",
    variables = vars_sel,
    n_vars = res_test$n_vars,
    accuracy = res_test$accuracy,
    balanced_accuracy = res_test$balanced_accuracy,
    auc = res_test$auc,
    recall1 = res_test$recall1,
    recall2 = res_test$recall2,
    score = res_test$score
  )
}


# ------------------------------------------------------------
# 7.5) SNB(MAP)
# ------------------------------------------------------------
# NO SE VA A IMPLEMENTAR


# ------------------------------------------------------------------------------
# 8. Validación cruzada externa: 10 ejecuciones de 10 folds con train/valid/test
# ------------------------------------------------------------------------------
# Finalmene, este bloque, organiza todo el experimento, guarda los resultados
# y calcula las medidas

# Para cada dataset:
# - Se preparan los datos.
# - Se crean 10 folds.

# En cada fold:
# - 1 fold se reserva como test externo;
# - Los 9 folds restantes se dividen en train y validación;

# se ejecutan los métodos;
# se guardan métricas, variables seleccionadas y tiempo.
# Se genera una tabla con todos los resultados.
# Se genera una tabla resumen por método.


# Creamos para un fold concreto los tres conjuntos: train, valid y test
crear_split_train_valid_test <- function(df, fold_test, folds) {
  
  idx_test <- folds[[fold_test]]    # Índices conjunto test
  resto <- setdiff(seq_len(nrow(df)), idx_test)
  
  # De las 9/10 partes restantes, usamos 2/3 train y 1/3 valid
  set.seed(1000 + fold_test)    # Fijamos semilla para que sea reproducible
  idx_train_rel <- caret::createDataPartition(
    df$Class[resto],
    p = 2/3,
    list = FALSE
  )
  
  # Los pasamos a índices reales del dataset
  idx_train <- resto[idx_train_rel]
  idx_valid <- setdiff(resto, idx_train)
  
  # Devuelve los tres datasets
  list(
    train = df[idx_train, , drop = FALSE],
    valid = df[idx_valid, , drop = FALSE],
    test  = df[idx_test,  , drop = FALSE]
  )
}

# Función que finalmente ejecuta el experimento completo para un dataset
run_experimento_dataset <- function(df, dataset_name,
                                    metric = "accuracy",
                                    n_folds = 10,
                                    n_reps = 10,
                                    combination_set = 25,
                                    incluir_boruta = TRUE,
                                    incluir_snb = TRUE) {
  
  df <- preparar_dataset(df, class_var = "Class")
  
  clase_1 <- names(which.min(table(df$Class)))    # Veo cual es la clase minoritaria
  
  # Para cada repetición vamos a generar nuevos folds (esto es clave!!)
  resultados <- list()
  contador <- 1
  
  for (rep in seq_len(n_reps)) {
    
    cat("\nDataset:", dataset_name, "| Repetición:", rep, "\n")
    
    # Fijamos semilla distinta en cada repetición (muy importante)
    set.seed(100 + rep)
    
    # Creamos los 10 folds
    folds <- caret::createFolds(
      df$Class,
      k = n_folds,
      list = TRUE,
      returnTrain = FALSE
    )
    
    # Para cada fold aplicamos cada uno de los métodos y gurdamos los resultados y los tiempos de ejecución
    for (fold in seq_len(n_folds)) {
      cat(
        Sys.time(), " Dataset:", dataset_name,
        " Rep:", rep,
        " Fold:", fold, "\n",
        file = paste0("resultados_v2/log_", dataset_name, ".txt"),
        append = TRUE
      )
      
      
      cat("   Fold:", fold, "\n")
      
      split_raw <- crear_split_train_valid_test(df, fold, folds)
      
      prep_disc <- ajustar_discretizador_mdlp(split_raw$train, class_var = "Class")
      split <- list(
        train = aplicar_discretizador_mdlp(split_raw$train, prep_disc),
        valid = aplicar_discretizador_mdlp(split_raw$valid, prep_disc),
        test  = aplicar_discretizador_mdlp(split_raw$test, prep_disc)
      )
      
      train_valid_raw <- dplyr::bind_rows(split_raw$train, split_raw$valid)
      prep_final <- ajustar_discretizador_mdlp(train_valid_raw, class_var = "Class")
      final_train <- aplicar_discretizador_mdlp(train_valid_raw, prep_final)
      final_test <- aplicar_discretizador_mdlp(split_raw$test, prep_final)
      
      # NB nuestro (Sparse Naïve Bayes)
      tiempo_sparse <- system.time({
        res_sparse <- fit_sparse_nb_fold(
          train = split$train,
          valid = split$valid,
          test  = split$test,
          metric = metric,
          combination_set = combination_set,
          clase_1 = clase_1,
          final_train = final_train,
          final_test = final_test
        )
      })
      
      # NB clásico
      tiempo_nb <- system.time({
        res_nb <- fit_classic_nb_fold(
          train = split$train,
          valid = split$valid,
          test  = split$test,
          metric = metric,
          clase_1 = clase_1,
          final_train = final_train,
          final_test = final_test
        )
      })
      
      # CFS
      tiempo_cfs <- system.time({
        res_cfs <- fit_cfs_nb_fold(
          train = split$train,
          valid = split$valid,
          test  = split$test,
          metric = metric,
          clase_1 = clase_1,
          final_train = final_train,
          final_test = final_test
        )
      })
      
      # Boruta
      if (incluir_boruta) {
        tiempo_boruta <- system.time({
        res_boruta <- fit_boruta_nb_fold(
          train = split$train,
          valid = split$valid,
            test  = split$test,
            metric = metric,
            clase_1 = clase_1,
            final_train = final_train,
            final_test = final_test
        )
        })
      } else {
        res_boruta <- NULL
        tiempo_boruta <- c(elapsed = 0)
      }
      
      # SNB
      if (incluir_snb) {
        tiempo_snb <- system.time({
        res_snb <- fit_snb_fold(
          train = split$train,
          valid = split$valid,
            test  = split$test,
            metric = metric,
            clase_1 = clase_1,
            final_train = final_train,
            final_test = final_test
        )
        })
      } else {
        res_snb <- NULL
        tiempo_snb <- c(elapsed = 0)
      }
      
      # Lista de los tiempos
      lista_fold <- list(
        res_sparse,
        res_nb,
        res_cfs,
        res_boruta,
        res_snb
      )
      
      tiempos <- c(
        tiempo_sparse["elapsed"],     #elapsed es el tiempo transcurrido en segundos
        tiempo_nb["elapsed"],
        tiempo_cfs["elapsed"],
        tiempo_boruta["elapsed"],
        tiempo_snb["elapsed"]
        # tiempo_snb_map["elapsed"]
      )
      
      
      # Guardamos todos los resultados obtenidos
      for (j in seq_along(lista_fold)) {
        
        res <- lista_fold[[j]]
        if (is.null(res)) next
        
        resultados[[contador]] <- data.frame(
          Dataset = dataset_name,
          Repetition = rep,   
          Fold = fold,
          Method = res$method,
          Metric_optimizada = metric,
          Accuracy = res$accuracy,
          BalancedAccuracy = res$balanced_accuracy,
          AUC = res$auc,
          Recall1 = res$recall1,
          Recall2 = res$recall2,
          Score = res$score,
          NumVariables = res$n_vars,
          Variables = paste(res$variables, collapse = ";"),
          TimeSeconds = as.numeric(tiempos[j]),
          stringsAsFactors = FALSE
        )
        
        contador <- contador + 1
      }
    }
  }
  
  # Cuando termina todos los folds, guardamos en una tabla
  resultados_df <- dplyr::bind_rows(resultados)
  
  write.csv2(
    resultados_df,
    file = paste0("resultados_v2/tablas/resultados_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  
  # Hacemos la tabla resumen
  resumen <- resultados_df %>%
    dplyr::group_by(Dataset, Method) %>%
    dplyr::summarise(
      Accuracy_media = mean(Accuracy, na.rm = TRUE),
      Accuracy_sd = sd(Accuracy, na.rm = TRUE),
      Recall1_media = mean(Recall1, na.rm = TRUE),
      Recall1_sd = sd(Recall1, na.rm = TRUE),
      Recall2_media = mean(Recall2, na.rm = TRUE),
      Recall2_sd = sd(Recall2, na.rm = TRUE),
      BalancedAccuracy_media = mean(BalancedAccuracy, na.rm = TRUE),
      BalancedAccuracy_sd = sd(BalancedAccuracy, na.rm = TRUE),
      AUC_media = mean(AUC, na.rm = TRUE),
      AUC_sd = sd(AUC, na.rm = TRUE),
      NumVariables_media = mean(NumVariables, na.rm = TRUE),
      NumVariables_sd = sd(NumVariables, na.rm = TRUE),
      Tiempo_medio = mean(TimeSeconds, na.rm = TRUE),
      Tiempo_sd = sd(TimeSeconds, na.rm = TRUE),
      .groups = "drop"
    )
  
  write.csv2(
    resumen,
    file = paste0("resultados_v2/tablas/resumen_", dataset_name, ".csv"),
    row.names = FALSE
  )
  
  list(resultados = resultados_df, resumen = resumen)   # Tb devuelvo las dos tablas en R
}


# EXTRA: Gráficos útiles para añadir en el TFG
# ------------------------------------------------------------
# 9. Figuras para el TFG
# ------------------------------------------------------------
# Ya tenemos los resultados, pero ahora lo que quiero es resumirlos visualmente

plot_resultados_dataset <- function(resultados_df, dataset_name) {
  
  # 1) Boxplot de accuracy
  p_acc <- ggplot(resultados_df, aes(x = Method, y = Accuracy)) +
    geom_boxplot() +
    labs(
      title = paste0(dataset_name, ": distribución de la accuracy en test"),
      x = NULL,
      y = "Accuracy"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_accuracy_boxplot.png"),
    plot = p_acc,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  
  # 2) Boxplot de número de variables
  p_vars <- ggplot(resultados_df, aes(x = Method, y = NumVariables)) +
    geom_boxplot() +
    labs(
      title = paste0(dataset_name, ": número de variables seleccionadas"),
      x = NULL,
      y = "Nº variables"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_variables_boxplot.png"),
    plot = p_vars,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  # 3) Trade-off accuracy vs variables
  p_tradeoff <- ggplot(
    resultados_df,
    aes(
      x = NumVariables,
      y = Accuracy,
      color = Method
    )
  ) +
    geom_jitter(width = 0.15, height = 0, alpha = 0.55, size = 2)  + # Separar un poco los puntos que caen encima enicma
    labs(
      title = paste0(dataset_name, ": compromiso accuracy-sparsity"),
      x = "Nº variables seleccionadas",
      y = "Accuracy",
      color = "Método"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5)
    )

  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_tradeoff.png"),
    plot = p_tradeoff,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  invisible(list(accuracy = p_acc, variables = p_vars, tradeoff = p_tradeoff))
}


# Extra: plot con medias
plot_medias <- function(resumen_df, dataset_name) {
  
  p <- ggplot(resumen_df, aes(x = Method, y = Accuracy_media)) +
    geom_col() +
    geom_errorbar(
      aes(
        ymin = Accuracy_media - Accuracy_sd,
        ymax = Accuracy_media + Accuracy_sd
      ),
      width = 0.2
    ) +
    labs(
      title = paste0(dataset_name, ": accuracy media ± desviación típica"),
      x = NULL,
      y = "Accuracy media"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  
  ggsave(
    paste0("resultados_v2/figuras/", dataset_name, "_accuracy_media.png"),
    p,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  p
}


# Plot del artículo:
plot_acc_sparsity_articulo <- function(resumen_df, dataset_name) {
  
  df_plot <- resumen_df %>%
    dplyr::mutate(
      ACC_pct = 100 * Accuracy_media,
      Metodo_label = paste0(Method, " (", round(Tiempo_medio, 2), "s)")
    )
  
  p <- ggplot(df_plot, aes(x = ACC_pct, y = NumVariables_media)) +
    geom_point(aes(shape = Method), size = 4) +
    ggrepel::geom_text_repel(
      aes(label = Metodo_label),
      size = 3.5,
      max.overlaps = Inf,
      box.padding = 0.4,
      point.padding = 0.3,
      segment.color = "grey60"
    ) +
    labs(
      title = paste0(dataset_name, " dataset"),
      x = "ACC (%)",
      y = "Sparsity"
    ) +
    scale_x_continuous(
      limits = c(
        min(df_plot$ACC_pct, na.rm = TRUE) - 2,
        max(df_plot$ACC_pct, na.rm = TRUE) + 3
      )
    ) +
    scale_y_continuous(
      limits = c(
        max(0, min(df_plot$NumVariables_media, na.rm = TRUE) - 1),
        max(df_plot$NumVariables_media, na.rm = TRUE) + 2
      )
    ) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    )
  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_ACC_vs_sparsity_articulo.png"),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  p
}


# Plot AUC vs sparsity (bd desbalanceadas)
plot_auc_sparsity_articulo <- function(resumen_df, dataset_name) {
  
  df_plot <- resumen_df %>%
    dplyr::mutate(
      AUC_pct = 100 * AUC_media,
      Metodo_label = paste0(Method, " (", round(Tiempo_medio, 2), "s)")
    )
  
  p <- ggplot(df_plot, aes(x = AUC_pct, y = NumVariables_media)) +
    geom_point(aes(shape = Method), size = 4) +
    ggrepel::geom_text_repel(
      aes(label = Metodo_label),
      size = 3.5,
      max.overlaps = Inf,
      box.padding = 0.4,
      point.padding = 0.3,
      segment.color = "grey60"
    ) +
    labs(
      title = paste0(dataset_name, " dataset"),
      x = "AUC (%)",
      y = "Sparsity"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    )
  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_AUC_vs_sparsity_articulo.png"),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  p
}

# Plot recall1 vs recall2 (bd desbalanceadas, optimizando AUC)
plot_recall1_recall2_articulo <- function(resumen_df, dataset_name) {
  
  if (!"Recall2_media" %in% names(resumen_df)) {
    resumen_df <- resumen_df %>%
      dplyr::mutate(Recall2_media = 2 * BalancedAccuracy_media - Recall1_media)
  }
  
  df_plot <- resumen_df %>%
    dplyr::mutate(
      Recall1_pct = 100 * Recall1_media,
      Recall2_pct = 100 * Recall2_media,
      Metodo_label = paste0(Method, " (", round(Tiempo_medio, 2), "s)")
    ) %>%
    dplyr::filter(!is.na(Recall1_pct), !is.na(Recall2_pct))
  
  margen <- 0.5
  x_lim <- c(
    max(0, floor(min(df_plot$Recall1_pct, na.rm = TRUE) - margen)),
    min(100, ceiling(max(df_plot$Recall1_pct, na.rm = TRUE) + margen))
  )
  y_lim <- c(
    max(0, floor(min(df_plot$Recall2_pct, na.rm = TRUE) - margen)),
    min(100, ceiling(max(df_plot$Recall2_pct, na.rm = TRUE) + margen))
  )
  
  p <- ggplot(df_plot, aes(x = Recall1_pct, y = Recall2_pct)) +
    geom_point(aes(shape = Method), size = 4) +
    ggrepel::geom_text_repel(
      aes(label = Metodo_label),
      size = 3.5,
      max.overlaps = Inf,
      box.padding = 0.4,
      point.padding = 0.3,
      segment.color = "grey60"
    ) +
    labs(
      title = paste0(dataset_name, " dataset"),
      x = "Recall1 (%)",
      y = "Recall2 (%)"
    ) +
    coord_equal(xlim = x_lim, ylim = y_lim) +
    scale_x_continuous(breaks = pretty(x_lim, n = 6)) +
    scale_y_continuous(breaks = pretty(y_lim, n = 6)) +
    theme_minimal(base_size = 13) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5)
    )
  
  ggsave(
    filename = paste0("resultados_v2/figuras/", dataset_name, "_Recall1_vs_Recall2_articulo.png"),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300
  )
  
  p
}

plot_recall1_sparsity_articulo <- function(resumen_df, dataset_name) {
  plot_recall1_recall2_articulo(resumen_df, dataset_name)
}

# ------------------------------------------------------------
# 10) Prueba inicial de algunos datasets por probar
# ------------------------------------------------------------
datasets <- list(
  WDBC = datos_wdbc,
  Wine = datos_wine,
  Mushroom = datos_mushroom,
  Waveform = datos_waveform,
  GermanCredit = datos_german,
  PageBlocks = datos_page,
  Australian = datos_australian,
  SpamBase = datos_spam,
  SPECTF = datos_spectf,
  IDLP = datos_IDLP,
  Shoppers = datos_shoppers

)

# PARA IR PROBANDO DATASETS (pero sin paralelizar --> :( tardaba mucho)
datasets_prueba <- datasets[c("Shoppers")]

resultados_prueba <- list()
resumenes_prueba <- list()

for (nm in names(datasets_prueba)) {
  
  out <- run_experimento_dataset(
    df = datasets_prueba[[nm]],
    dataset_name = nm,
    metric = "accuracy",
    n_folds = 10,
    combination_set = 25
  )
  
  resultados_prueba[[nm]] <- out$resultados
  resumenes_prueba[[nm]] <- out$resumen
  
  plot_resultados_dataset(out$resultados, nm)
  plot_medias(out$resumen, nm)
  plot_acc_sparsity_articulo(out$resumen, nm)
}

resultados_prueba_finales <- dplyr::bind_rows(resultados_prueba)
resumen_prueba_final <- dplyr::bind_rows(resumenes_prueba)

write.csv2(resultados_prueba_finales, "resultados_v2/tablas/resultados_prueba.csv", row.names = FALSE)
write.csv2(resumen_prueba_final, "resultados_v2/tablas/resumen_prueba.csv", row.names = FALSE)



# EXTRA: SI ENCIMA ES BD DESBALANCEADA CARGAR TAMBIÉN ESTO
resultados_desbalanceados <- list()
resumenes_desbalanceados <- list()

# AUC
for (nm in names(datasets_prueba)) {
  
  out <- run_experimento_dataset(
    df = datasets_prueba[[nm]],
    dataset_name = nm,
    metric = "auc",
    n_folds = 10,
    n_reps = 10,
    combination_set = 25
  )
  
  resultados_desbalanceados[[nm]] <- out$resultados
  resumenes_desbalanceados[[nm]] <- out$resumen
  plot_auc_sparsity_articulo(out$resumen, nm)
  plot_recall1_recall2_articulo(out$resumen, nm)
}

resultados_desbalanceados_final <- dplyr::bind_rows(resultados_desbalanceados)
resumen_desbalanceados_final <- dplyr::bind_rows(resumenes_desbalanceados)

write.csv2(
  resultados_desbalanceados_final,
  "resultados_v2/tablas/resultados_desbalanceados_auc.csv",
  row.names = FALSE
)

write.csv2(
  resumen_desbalanceados_final,
  "resultados_v2/tablas/resumen_desbalanceados_auc.csv",
  row.names = FALSE
)

# EXTENSIONES: 
# Probar con otras métricas 
# si optimizo recall_1 va a castigar mucho a las otras clases
# por lo que maximizo Recall1, pero solo aceptar subconjuntos donde Recall2 ≥ 0.60

# resultados_desbalanceados <- list()
# resumenes_desbalanceados <- list()
# 
# for (nm in names(datasets_prueba)) {
#   
#   out <- run_experimento_dataset(
#     df = datasets_prueba[[nm]],
#     dataset_name = nm,
#     metric = "recall1_cond_recall2_0.60",
#     n_folds = 10,
#     n_reps = 10,
#     combination_set = 25
#   )
#   
#   resultados_desbalanceados[[nm]] <- out$resultados
#   resumenes_desbalanceados[[nm]] <- out$resumen
#   plot_auc_sparsity_articulo(out$resumen, nm)
#   plot_recall1_recall2_articulo(out$resumen, nm)
# }
# 
# 
# resultados_desbalanceados_final <- dplyr::bind_rows(resultados_desbalanceados)
# resumen_desbalanceados_final <- dplyr::bind_rows(resumenes_desbalanceados)
# 
# write.csv2(
#   resultados_desbalanceados_final,
#   "resultados_v2/tablas/resultados_desbalanceados_recall1_cond_recall2_0_60.csv",
#   row.names = FALSE
# )
# 
# write.csv2(
#   resumen_desbalanceados_final,
#   "resultados_v2/tablas/resumen_desbalanceados_recall1_cond_recall2_0_60.csv",
#   row.names = FALSE
# )


