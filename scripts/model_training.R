# ============================================
# Cyclone Intensity Prediction - Model Training
# (WITH MODEL SAVING & REUSE)
# ============================================

library(readr)
library(dplyr)
library(randomForest)
library(e1071)
library(FNN)
library(xgboost)
library(keras)

# -------------------------------
# 1. Load Data
# -------------------------------
train <- read_csv("D:/Rcyclone_project/data/selected/train_selected.csv")
test  <- read_csv("D:/Rcyclone_project/data/selected/test_selected.csv")

train_y <- train$msl
test_y  <- test$msl

train_x <- train %>% select(-msl)
test_x  <- test %>% select(-msl)

# -------------------------------
# 2. Evaluation Function
# -------------------------------
evaluate_model <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  mae  <- mean(abs(actual - predicted))
  r2   <- cor(actual, predicted)^2
  
  return(c(RMSE = rmse, MAE = mae, R2 = r2))
}

results <- list()

# Create models folder if not exists
dir.create("D:/Rcyclone_project/models", showWarnings = FALSE)

# -------------------------------
# 3. RANDOM FOREST
# -------------------------------
rf_path <- "D:/Rcyclone_project/models/rf_model.rds"

if (file.exists(rf_path)) {
  rf_model <- readRDS(rf_path)
  cat("Loaded RF model\n")
} else {
  rf_model <- randomForest(x = train_x, y = train_y, ntree = 100)
  saveRDS(rf_model, rf_path)
  cat("Trained & Saved RF model\n")
}

rf_pred <- predict(rf_model, test_x)
results$RandomForest <- evaluate_model(test_y, rf_pred)

write.csv(rf_pred, "D:/Rcyclone_project/models/rf_pred.csv")

# -------------------------------
# 4. SVR
# -------------------------------
svr_path <- "D:/Rcyclone_project/models/svr_model.rds"

if (file.exists(svr_path)) {
  svr_model <- readRDS(svr_path)
  cat("Loaded SVR model\n")
} else {
  svr_model <- svm(x = train_x, y = train_y)
  saveRDS(svr_model, svr_path)
  cat("Trained & Saved SVR model\n")
}

svr_pred <- predict(svr_model, test_x)
results$SVR <- evaluate_model(test_y, svr_pred)

write.csv(svr_pred, "D:/Rcyclone_project/models/svr_pred.csv")

# -------------------------------
# 5. KNN
# -------------------------------
train_x_mat <- as.matrix(train_x)
test_x_mat  <- as.matrix(test_x)

knn_pred <- knn.reg(train_x_mat, test_x_mat, y = train_y, k = 5)$pred
results$KNN <- evaluate_model(test_y, knn_pred)

write.csv(knn_pred, "D:/Rcyclone_project/models/knn_pred.csv")

cat("KNN Done\n")

# -------------------------------
# 6. XGBOOST
# -------------------------------
xgb_path <- "D:/Rcyclone_project/models/xgb_model.model"

train_matrix <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_matrix  <- xgb.DMatrix(data = as.matrix(test_x),  label = test_y)

if (file.exists(xgb_path)) {
  xgb_model <- xgb.load(xgb_path)
  cat("Loaded XGBoost model\n")
} else {
  params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
  
  xgb_model <- xgb.train(
    params = params,
    data = train_matrix,
    nrounds = 100,
    verbose = 0
  )
  
  xgb.save(xgb_model, xgb_path)
  cat("Trained & Saved XGBoost model\n")
}

xgb_pred <- predict(xgb_model, test_matrix)
results$XGBoost <- evaluate_model(test_y, xgb_pred)

write.csv(xgb_pred, "D:/Rcyclone_project/models/xgb_pred.csv")

# -------------------------------
# 7. LSTM (KERAS)
# -------------------------------
lstm_path <- "D:/Rcyclone_project/models/lstm_model.h5"

create_sequences <- function(data, target, time_steps = 3) {
  X <- list()
  y <- c()
  
  for (i in 1:(nrow(data) - time_steps)) {
    X[[i]] <- data[i:(i + time_steps - 1), ]
    y[i] <- target[i + time_steps]
  }
  
  X <- array(unlist(X), dim = c(length(X), time_steps, ncol(data)))
  return(list(X = X, y = y))
}

train_seq <- create_sequences(as.matrix(train_x), train_y, 3)
test_seq  <- create_sequences(as.matrix(test_x), test_y, 3)

if (file.exists(lstm_path)) {
  lstm_model <- load_model_hdf5(lstm_path)
  cat("Loaded LSTM model\n")
} else {
  lstm_model <- keras_model_sequential() %>%
    layer_lstm(units = 50, input_shape = c(3, ncol(train_x))) %>%
    layer_dense(units = 1)
  
  lstm_model %>% compile(
    loss = "mse",
    optimizer = "adam"
  )
  
  lstm_model %>% fit(
    train_seq$X, train_seq$y,
    epochs = 5,
    batch_size = 32,
    verbose = 0
  )
  
  save_model_hdf5(lstm_model, lstm_path)
  cat("Trained & Saved LSTM model\n")
}

lstm_pred <- lstm_model %>% predict(test_seq$X)

results$LSTM <- evaluate_model(test_seq$y, lstm_pred)

write.csv(lstm_pred, "D:/Rcyclone_project/models/lstm_pred.csv")

# -------------------------------
# 8. Save Metrics
# -------------------------------
results_df <- do.call(rbind, results)
print(results_df)

write.csv(results_df, "D:/Rcyclone_project/outputs/metrics/model_results.csv")

cat("All Models Completed & Saved\n")