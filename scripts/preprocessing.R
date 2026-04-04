# ============================================
# Cyclone Intensity Prediction - Preprocessing
# (Paper-Aligned + Performance Optimized)
#This preprocessing code prepares the ERA5 dataset for cyclone intensity prediction by:

#Cleaning the data by handling missing values and removing highly incomplete features
#Maintaining time-series integrity by sorting data and splitting it monthly (15/10/remaining days)
#Creating useful features like wind speed and a lagged pressure variable
#Standardizing features using training data to avoid data leakage
#Producing final train, test, and validation datasets ready for machine learning models
# ============================================

library(readr)
library(dplyr)
library(lubridate)
library(caret)
library(zoo)

# -------------------------------
# 1. Load Dataset
# -------------------------------
file_path <- "D:/Rcyclone_project/data/raw/indian_era5.csv"
data <- read_csv(file_path)

cat("Initial Shape:", dim(data), "\n")

# -------------------------------
# 2. Sort Data (Time-Series Integrity)
# -------------------------------
data <- data %>%
  arrange(date, latitude, longitude)

# -------------------------------
# 3. Handle Missing Values
# -------------------------------

missing_percent <- sapply(data, function(x) sum(is.na(x)) / length(x)) * 100
print(sort(missing_percent, decreasing = TRUE))

# Remove columns with >50% missing
threshold <- 50
cols_to_keep <- names(missing_percent[missing_percent < threshold])
data <- data[, cols_to_keep]

cat("After removing high-missing columns:", dim(data), "\n")

# Interpolation (spatio-temporal)
data <- data %>%
  group_by(latitude, longitude) %>%
  mutate(across(where(is.numeric), ~ na.approx(., na.rm = FALSE))) %>%
  ungroup()

# Fill remaining NA with median
data <- data %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

cat("Missing values handled.\n")

# -------------------------------
# 4. Feature Engineering (SAFE ADDITIONS)
# -------------------------------

# Wind speed (important derived feature)
data <- data %>%
  mutate(wind_speed = sqrt(u10^2 + v10^2))

# Lag feature (only 1 to stay aligned)
data <- data %>%
  group_by(latitude, longitude) %>%
  mutate(msl_lag1 = lag(msl, 1)) %>%
  ungroup()

# Remove rows with NA due to lag
data <- na.omit(data)

# -------------------------------
# 5. Extract Time Components
# -------------------------------
data <- data %>%
  mutate(
    year  = year(date),
    month = month(date),
    day   = day(date)
  )

# -------------------------------
# 6. Time-Based Split (PAPER METHOD)
# -------------------------------
# 1–15 → Train
# 16–25 → Test
# 26–end → Validation

train_data <- data %>% filter(day >= 1 & day <= 15)
test_data  <- data %>% filter(day >= 16 & day <= 25)
val_data   <- data %>% filter(day >= 26)

cat("Train:", nrow(train_data), "\n")
cat("Test:", nrow(test_data), "\n")
cat("Validation:", nrow(val_data), "\n")

# -------------------------------
# 7. Separate Target Variable
# -------------------------------
# Target = msl

train_y <- train_data$msl
test_y  <- test_data$msl
val_y   <- val_data$msl

train_x <- train_data %>% select(-msl, -date)
test_x  <- test_data %>% select(-msl, -date)
val_x   <- val_data %>% select(-msl, -date)

# -------------------------------
# 8. Normalization (NO DATA LEAKAGE)
# -------------------------------
# Fit only on training data

preProc <- preProcess(train_x, method = c("center", "scale"))

train_x_scaled <- predict(preProc, train_x)
test_x_scaled  <- predict(preProc, test_x)
val_x_scaled   <- predict(preProc, val_x)

cat("Scaling completed.\n")

# -------------------------------
# 9. Combine Back
# -------------------------------
train_final <- cbind(train_x_scaled, msl = train_y)
test_final  <- cbind(test_x_scaled,  msl = test_y)
val_final   <- cbind(val_x_scaled,   msl = val_y)

# -------------------------------
# 10. Save Files
# -------------------------------
write_csv(train_final, "D:/Rcyclone_project/data/processed/train.csv")
write_csv(test_final,  "D:/Rcyclone_project/data/processed/test.csv")
write_csv(val_final,   "D:/Rcyclone_project/data/processed/val.csv")

cat("Preprocessing Completed Successfully.\n")