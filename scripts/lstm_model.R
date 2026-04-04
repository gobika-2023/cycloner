# ============================================
# Cyclone Intensity Prediction - LSTM Model
# ============================================

library(readr)
library(dplyr)
library(keras)

# -------------------------------
# 1. Load Data
# -------------------------------
train <- read_csv("D:/Rcyclone_project/data/selected/train_selected.csv")
test  <- read_csv("D:/Rcyclone_project/data/selected/test_selected.csv")

train_y <- train$msl
test_y  <- test$msl

train_x <- as.matrix(train %>% select(-msl))
test_x  <- as.matrix(test %>% select(-msl))

# -------------------------------
# 2. Create Sequences
# -------------------------------
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

time_steps <- 3

train_seq <- create_sequences(train_x, train_y, time_steps)
test_seq  <- create_sequences(test_x, test_y, time_steps)

# -------------------------------
# 3. Model Path
# -------------------------------
model_path <- "D:/Rcyclone_project/models/lstm_model.h5"

dir.create("D:/Rcyclone_project/models", showWarnings = FALSE)

# -------------------------------
# 4. Load or Train Model
# -------------------------------
if (file.exists(model_path)) {
  
  model <- load_model_hdf5(model_path)
  cat("Loaded existing LSTM model\n")
  
} else {
  
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64, input_shape = c(time_steps, ncol(train_x))) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = "adam"
  )
  
  model %>% fit(
    train_seq$X, train_seq$y,
    epochs = 8,
    batch_size = 32,
    validation_split = 0.2,
    verbose = 1
  )
  
  save_model_hdf5(model, model_path)
  cat("Trained & Saved LSTM model\n")
}

# -------------------------------
# 5. Prediction
# -------------------------------
pred <- model %>% predict(test_seq$X)

# -------------------------------
# 6. Evaluation
# -------------------------------
rmse <- sqrt(mean((test_seq$y - pred)^2))
mae  <- mean(abs(test_seq$y - pred))
r2   <- cor(test_seq$y, pred)^2

cat("\n===== LSTM RESULTS =====\n")
cat("RMSE:", rmse, "\n")
cat("MAE :", mae, "\n")
cat("R²  :", r2, "\n")

# -------------------------------
# 7. Save Predictions
# -------------------------------
write.csv(pred, "D:/Rcyclone_project/models/lstm_pred.csv")

# -------------------------------
# 8. Save Metrics
# -------------------------------
metrics <- data.frame(RMSE = rmse, MAE = mae, R2 = r2)

write.csv(metrics, "D:/Rcyclone_project/outputs/metrics/lstm_results.csv")

cat("LSTM Completed Successfully\n")