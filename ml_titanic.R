# Pakete
suppressPackageStartupMessages({
  library(xgboost)
  library(Matrix)
  library(caret)
  library(pROC)
})

set.seed(42)

# 1) Daten laden, falls df (mit survived) nicht im Workspace ist
if (!exists("df")) {
  df <- read.csv("bereinigter_titanic_datensatz.csv")
}

# 2) Sicherstellen, dass df_features vorhanden ist (ohne survived)
if (!exists("df_features")) {
  df_features <- subset(df, select = -survived)
}

# 3) Zielvariable y aus dem "richtigen" Datensatz (Original enthält survived)
y <- df$survived
y <- as.numeric(y) # xgboost erwartet numerisch (0/1)

# 4) Feature-Selektion: nur numerische/logische Spalten für XGBoost
#    (Objektspalten wie name/ticket/cabin/boat/deck fliegen raus)
is_num_or_logical <- sapply(df_features, function(col) is.numeric(col) || is.logical(col))
X <- df_features[ , is_num_or_logical, drop = FALSE]

# 5) Logicals -> numerisch (TRUE/FALSE -> 1/0)
for (cl in names(X)) {
  if (is.logical(X[[cl]])) X[[cl]] <- as.numeric(X[[cl]])
}

# 6) Fehlende Werte (falls vorhanden) durch Median ersetzen (robust, einfach)
train_medians <- vapply(X, function(v) if (all(is.na(v))) NA_real_ else suppressWarnings(median(v, na.rm = TRUE)), numeric(1))
for (cl in names(X)) {
  if (anyNA(X[[cl]])) {
    med <- train_medians[[cl]]
    if (is.na(med)) med <- 0
    X[[cl]][is.na(X[[cl]])] <- med
  }
}

cat("\n[Check] Verwendete Feature-Spalten:\n")
print(colnames(X))

# 7) Train/Test-Split (stratifiziert nach y)
idx_train <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[idx_train, , drop = FALSE]
X_test  <- X[-idx_train, , drop = FALSE]
y_train <- y[idx_train]
y_test  <- y[-idx_train]

# 8) In DMatrix umwandeln
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test)

# 9) XGBoost-Parameter (solide Startwerte)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 5,
  subsample = 0.9,
  colsample_bytree = 0.9,
  min_child_weight = 1
)

# 10) Training mit Early Stopping
watchlist <- list(train = dtrain, eval = dtest)
bst <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 2000,
  watchlist = watchlist,
  early_stopping_rounds = 50,
  verbose = 1
)

# 11) Evaluation
pred_prob <- predict(bst, dtest)
pred_cls  <- ifelse(pred_prob >= 0.5, 1, 0)

acc <- mean(pred_cls == y_test)
roc_obj <- roc(response = y_test, predictor = pred_prob)
auc_val <- as.numeric(auc(roc_obj))

cat(sprintf("Accuracy: %.3f\nAUC: %.3f\n", acc, auc_val))

# Confusion Matrix
cat("\nKonfusionsmatrix (Test-Set):\n")
print(table(Pred = pred_cls, True = y_test))

# --- Ergebnis in Prozent & kompaktes Fazit ausgeben ---
acc_pct <- round(100 * acc, 2)
auc_pct <- round(100 * auc_val, 2)

cat(sprintf("\n--- MODELLERGEBNIS ---\n"))
cat(sprintf("Trefferquote (Accuracy): %0.2f%%\n", acc_pct))
cat(sprintf("AUC (Fläche unter ROC): %0.2f%%\n", auc_pct))
cat("Konfusionsmatrix (Test-Set):\n")
print(table(Pred = pred_cls, True = y_test))

# Kurzes Fazit
baseline <- max(mean(y_test == 1), mean(y_test == 0)) # naive Mehrheitsklasse
baseline_pct <- round(100 * baseline, 2)
cat(sprintf(
  "\nFazit: Das XGBoost-Modell erreicht %0.2f%% Accuracy (Baseline: %0.2f%%) und %0.2f%% AUC.\n",
  acc_pct, baseline_pct, auc_pct
))

# --- FEATURE IMPORTANCE -----------------------------------------------------
importance <- xgb.importance(model = bst, feature_names = colnames(X_train))

cat("\n--- WICHTIGSTE MERKMALE FÜR DAS ÜBERLEBEN ---\n")
# NAs entfernen (falls vorhanden) und Top-N zeigen
importance <- na.omit(importance)
if (nrow(importance) == 0) {
  cat("Keine Importance-Werte verfügbar. Prüfe die Feature-Matrix X_train.\n")
} else {
  print(importance[1:min(10, nrow(importance)), c("Feature", "Gain", "Cover", "Frequency")])
}

# Optional: Visualisierung (xgboost intern)
if (requireNamespace("xgboost", quietly = TRUE) && nrow(importance) > 0) {
  try({
    xgb.plot.importance(importance, top_n = min(10, nrow(importance)),
                        measure = "Gain",
                        rel_to_first = TRUE,
                        main = "Einfluss der Merkmale auf Überleben")
  }, silent = TRUE)
}

# --- SHAP-ANALYSE (ohne Zusatzpakete) --------------------------------------
# Wir nutzen xgboost::predict(..., predcontrib=TRUE).
# Hinweis: Letzte Spalte ist der Bias-Term -> entfernen.

shap_compute <- function(model, X_mat) {
  shap <- predict(model, X_mat, predcontrib = TRUE)
  # Bias-Spalte entfernen
  shap <- shap[, -ncol(shap), drop = FALSE]
  shap
}

# SHAP auf Train- und Test-Set berechnen
X_train_mat <- as.matrix(X_train)
X_test_mat  <- as.matrix(X_test)

shap_train <- shap_compute(bst, X_train_mat)
shap_test  <- shap_compute(bst, X_test_mat)

# Globaler Überblick: mittlere |SHAP|-Beiträge pro Feature (Train)
shap_mean_abs <- colMeans(abs(shap_train))
shap_summary  <- data.frame(
  Feature = colnames(X_train),
  MeanAbsSHAP = shap_mean_abs[match(colnames(X_train), names(shap_mean_abs))]
)
shap_summary <- shap_summary[order(-shap_summary$MeanAbsSHAP), ]

cat("\n--- SHAP Global (mittlere |Beitragswerte|) ---\n")
print(head(shap_summary, 10), row.names = FALSE)

# Einfache Importance-Plot-Alternative (barplot, Top 10)
top_n <- min(10, nrow(shap_summary))
op <- par(mar = c(5, 10, 3, 2))
barplot(
  shap_summary$MeanAbsSHAP[1:top_n][rev(seq_len(top_n))],
  names.arg = shap_summary$Feature[1:top_n][rev(seq_len(top_n))],
  horiz = TRUE,
  las = 1,
  main = "SHAP (Mean |Contribution|) – Top Features"
)
par(op)

# Dependence-Plot (Basis-Variante) für die Top-Features
plot_shap_dependence <- function(feature_name, X_mat, shap_mat, color_by = NULL) {
  if (!feature_name %in% colnames(X_mat)) return(invisible(NULL))
  x_vals   <- X_mat[, feature_name]
  shap_vals <- shap_mat[, feature_name]
  # Farbkodierung optional über ein zweites Feature
  if (!is.null(color_by) && color_by %in% colnames(X_mat)) {
    col_vals <- X_mat[, color_by]
    # einfache kontinuierliche Farbskala
    cols <- grDevices::gray((col_vals - min(col_vals, na.rm=TRUE)) /
                              (max(col_vals, na.rm=TRUE) - min(col_vals, na.rm=TRUE)))
  } else {
    cols <- "black"
  }
  plot(x_vals, shap_vals,
       pch = 16, cex = 0.7, col = cols,
       xlab = feature_name, ylab = paste0("SHAP(", feature_name, ")"),
       main = paste("SHAP Dependence –", feature_name))
  abline(h = 0, lty = 2)
}

# Dependence-Plots für die Top 3 Features (mit optionaler Farb-Interaktion)
top_feats <- head(shap_summary$Feature, 3)
for (f in top_feats) {
  by_feat <- setdiff(top_feats, f)
  by_feat <- if (length(by_feat) > 0) by_feat[1] else NULL
  plot_shap_dependence(f, X_train_mat, shap_train, color_by = by_feat)
}


