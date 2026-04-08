cat("Starting Central Surface Pressure Prediction...\n")

# ---------------------------
# LOAD LIBRARIES
# ---------------------------
packages <- c("dplyr","readr","ggplot2","caret")
for (p in packages) {
  if (!require(p, character.only = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}

# ---------------------------
# LOAD DATA
# ---------------------------
data <- read_csv("D:/final_era5.csv", show_col_types = FALSE)
names(data) <- tolower(names(data))

# ---------------------------
# CLEAN DATA
# ---------------------------
data <- data %>%
  mutate(date = as.Date(date)) %>%
  mutate(across(-date, as.numeric)) %>%
  filter(!is.na(msl))

# ---------------------------
# REDUCE SIZE (IMPORTANT 🔥)
# ---------------------------
set.seed(123)
data <- data %>% sample_n(50000)   # 🔥 reduces from 533k → 50k

# ---------------------------
# REMOVE DATE
# ---------------------------
data_model <- data %>% select(-date)

# ---------------------------
# SPLIT
# ---------------------------
train_index <- createDataPartition(data_model$msl, p = 0.8, list = FALSE)

train_data <- data_model[train_index, ]
test_data  <- data_model[-train_index, ]

# ---------------------------
# FAST MODEL (LINEAR REGRESSION)
# ---------------------------
cat("Training fast model...\n")

model <- train(
  msl ~ .,
  data = train_data,
  method = "lm"   # 🔥 FAST (no stuck)
)

cat("Training done\n")

# ---------------------------
# PREDICTION
# ---------------------------
pred <- predict(model, test_data)
actual <- test_data$msl

# ---------------------------
# METRICS
# ---------------------------
rmse <- sqrt(mean((pred - actual)^2))
mae  <- mean(abs(pred - actual))
r2   <- cor(pred, actual)^2

cat("\n===== PERFORMANCE =====\n")
cat("RMSE :", rmse, "\n")
cat("MAE  :", mae, "\n")
cat("R²   :", r2, "\n")

# ---------------------------
# PLOT
# ---------------------------
results <- data.frame(Actual = actual, Predicted = pred)

ggplot(results, aes(Actual, Predicted)) +
  geom_point(alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0) +
  theme_minimal() +
  labs(title = "Actual vs Predicted Pressure")

cat("\n✅ DONE (FAST)\n")
