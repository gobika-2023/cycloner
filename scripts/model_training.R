# ============================================
# Cyclone Intensity Prediction - FINAL OPTIMIZED
# (Paper-Aligned + R² Boost Version)
# ============================================

library(readr)
library(dplyr)
library(xgboost)
library(caret)
library(ggplot2)

cat("🚀 Loading selected dataset...\n")

# -------------------------------
# 0. Create Folders
# -------------------------------
dir.create("D:/Rcyclone_project/models", showWarnings = FALSE)
dir.create("D:/Rcyclone_project/outputs/metrics", showWarnings = FALSE)
dir.create("D:/Rcyclone_project/outputs/plots", showWarnings = FALSE)

# -------------------------------
# 1. Load Data
# -------------------------------
train <- read_csv("D:/Rcyclone_project/data/selected/train_selected.csv", show_col_types = FALSE)
test  <- read_csv("D:/Rcyclone_project/data/selected/test_selected.csv", show_col_types = FALSE)
val   <- read_csv("D:/Rcyclone_project/data/selected/val_selected.csv", show_col_types = FALSE)

# -------------------------------
# 2. Advanced Feature Engineering
# -------------------------------
feature_engineering <- function(df) {
  df %>%
    mutate(
      wind_energy = (u10^2 + v10^2),
      wind_direction = atan2(v10, u10),
      pressure_change = msl - lag(msl, 1, default = first(msl)),
      interaction_1 = u10 * v10,
      interaction_2 = wind_speed * msl_lag1
    )
}

train <- feature_engineering(train)
test  <- feature_engineering(test)
val   <- feature_engineering(val)

# -------------------------------
# 3. Split Features/Target
# -------------------------------
train_y <- train$msl
test_y  <- test$msl
val_y   <- val$msl

train_x <- as.matrix(train %>% select(-msl))
test_x  <- as.matrix(test %>% select(-msl))
val_x   <- as.matrix(val %>% select(-msl))

train_matrix <- xgb.DMatrix(data = train_x, label = train_y)
val_matrix   <- xgb.DMatrix(data = val_x, label = val_y)
test_matrix  <- xgb.DMatrix(data = test_x, label = test_y)

# -------------------------------
# 4. Hyperparameter Tuning (CV)
# -------------------------------
cat("🔍 Running Cross Validation...\n")

params <- list(
  objective = "reg:squarederror",
  eta = 0.01,
  max_depth = 5,
  subsample = 0.7,
  colsample_bytree = 0.7,
  min_child_weight = 5,
  gamma = 1,
  lambda = 2,
  alpha = 1,
  eval_metric = "rmse"
)

cv_model <- xgb.cv(
  params = params,
  data = train_matrix,
  nrounds = 1500,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = 0,
  maximize = FALSE
)

best_nrounds <- cv_model$best_iteration
if (is.null(best_nrounds) || best_nrounds < 1) best_nrounds <- 500
cat("✅ Best rounds:", best_nrounds, "\n")

# -------------------------------
# 5. Final Model Training
# -------------------------------
cat("🚀 Training final optimized XGBoost...\n")

xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = best_nrounds,
  evals = list(train = train_matrix, val = val_matrix),  # fixed evals
  early_stopping_rounds = 50,
  verbose = 1
)

# Save model
xgb.save(xgb_model, "D:/Rcyclone_project/models/xgb_model_optimized.model")

# -------------------------------
# 6. Predictions
# -------------------------------
train_pred <- predict(xgb_model, train_matrix)
val_pred   <- predict(xgb_model, val_matrix)
test_pred  <- predict(xgb_model, test_matrix)

# -------------------------------
# 7. Evaluation
# -------------------------------
evaluate <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  r2   <- cor(actual, pred)^2
  return(c(RMSE = rmse, MAE = mae, R2 = r2))
}

results <- rbind(
  Train = evaluate(train_y, train_pred),
  Validation = evaluate(val_y, val_pred),
  Test = evaluate(test_y, test_pred)
)
print(results)
write.csv(results, "D:/Rcyclone_project/outputs/metrics/final_results_optimized.csv")

# -------------------------------
# 8. Confusion Matrix (5 Classes)
# -------------------------------
categorize <- function(x) {
  cut(x,
      breaks = quantile(x, probs = seq(0,1,0.2), na.rm = TRUE),
      labels = c("Very Low","Low","Medium","High","Very High"),
      include.lowest = TRUE)
}

actual_cat <- categorize(test_y)
pred_cat   <- categorize(test_pred)

conf_matrix <- caret::confusionMatrix(pred_cat, actual_cat)
write.csv(conf_matrix$table,
          "D:/Rcyclone_project/outputs/metrics/confusion_matrix_optimized.csv")

# -------------------------------
# 9. Graphs
# -------------------------------
df_plot <- data.frame(actual = test_y, predicted = test_pred)
df_plot$residuals <- df_plot$actual - df_plot$predicted

# Actual vs Predicted
p1 <- ggplot(df_plot, aes(actual, predicted)) +
  geom_point(alpha=0.3, color="blue") +
  geom_smooth(method="lm", color="red") +
  ggtitle("Actual vs Predicted (Optimized Model)") +
  theme_minimal()
ggsave("D:/Rcyclone_project/outputs/plots/actual_vs_pred_optimized.png", p1)

# Residual Plot
p2 <- ggplot(df_plot, aes(predicted, residuals)) +
  geom_point(alpha=0.3) +
  geom_hline(yintercept = 0, color="red") +
  ggtitle("Residual Plot") +
  theme_minimal()
ggsave("D:/Rcyclone_project/outputs/plots/residual_plot.png", p2)

# Feature Importance
importance_matrix <- xgb.importance(model = xgb_model)
png("D:/Rcyclone_project/outputs/plots/feature_importance_optimized.png")
xgb.plot.importance(importance_matrix)
dev.off()

cat("✅ FINAL OPTIMIZED MODEL COMPLETE\n")