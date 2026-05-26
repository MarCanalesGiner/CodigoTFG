# Código del Trabajo Fin de Grado: Sparse Naïve Bayes

Este repositorio contiene el código desarrollado en R para el Trabajo Fin de Grado **Selección de variables para el clasificador Naïve Bayes**. El objetivo del proyecto es estudiar e implementar un clasificador Naïve Bayes disperso, basado en la estructura de dependencia entre variables, y compararlo con otros métodos de selección de variables.

## Contenido del repositorio
Los scripts principales son:

- `Datos.R`: carga los conjuntos de datos utilizados en el trabajo y genera una tabla resumen con sus principales características: número de observaciones, número de clases, número de variables continuas y categóricas, y distribución de las clases.

- `descripcion_datasets.R`: genera una descripción adicional de algunos conjuntos de datos y mapas de calor basados en información mutua para visualizar la estructura de dependencia entre variables.

- `comp_medidas_dependencia_M.R`: compara distintas medidas de dependencia entre variables, como Pearson, Spearman, Hoeffding, correlación de distancias e información mutua. También genera mapas de calor de las matrices de dependencia.

- `Codigo_TFG_metodos.R`: contiene la implementación principal de los experimentos. Incluye la construcción de la matriz de dependencia, la generación de subconjuntos candidatos, el clasificador Sparse Naïve Bayes y la comparación con Naïve Bayes clásico, CFS, Boruta y SNB.

## Conjuntos de datos
Los experimentos se realizan sobre distintos conjuntos de datos procedentes principalmente del repositorio UCI Machine Learning Repository y del paquete `mlbench`:

- Wine
- Mushroom
- Waveform
- WDBC
- Australian Credit
- Page Blocks
- Indian Liver Patient Dataset (IDLP)
- Online Shoppers Purchasing Intention Dataset (Shoppers)

## Paquetes necesarios
Para ejecutar el código es necesario tener instalados los siguientes paquetes de R:

```r
install.packages(c(
  "ucimlrepo", "mlbench", "rattle", "caret", "e1071", "klaR",
  "infotheo", "discretization", "dplyr", "tidyr", "purrr",
  "ggplot2", "reshape2", "gridExtra", "FSelector", "Boruta",
  "randomForest", "pROC", "ggrepel", "Hmisc", "energy", "knitr"
))
