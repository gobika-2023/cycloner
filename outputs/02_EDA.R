# =========================
# LOAD DATA
# =========================
data <- read.csv("data/cleaned/final_era5.csv")

# Fix columns
data$cape <- as.numeric(as.character(data$cape))
data$tp   <- as.numeric(as.character(data$tp))
data$msl  <- as.numeric(as.character(data$msl))
data$date <- as.Date(data$date, format="%Y-%m-%d")

data <- na.omit(data)

# =========================
# CREATE FOLDER
# =========================
dir.create("eda", showWarnings = FALSE)
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)

# =========================
# GRAPH 1: CAPE PIE
# =========================
data$category <- cut(data$cape, breaks=5,
                     labels=c("Low","Moderate","High","Very High","Extreme"))

counts <- table(data$category)
counts <- counts[counts > 0]

png("eda/graph1_cape_pie.png",800,600)
pie(counts, col=rainbow(length(counts)))
dev.off()

# =========================
# GRAPH 2: MSL TREND
# =========================
data$month <- format(data$date,"%Y-%m")
monthly <- aggregate(data$msl, by=list(data$month), FUN=mean)

png("eda/graph2_monthly_msl.png",800,600)
plot(monthly$x, type="l", col="blue")
dev.off()

# =========================
# GRAPH 3: TP vs MEAN
# =========================
mean_tp <- mean(data$tp)

png("eda/graph3_tp_vs_mean.png",800,600)
plot(data$date, data$tp, type="l", col="green")
abline(h=mean_tp, col="red")
dev.off()

# =========================
# LOAD PREDICTIONS
# =========================
raw <- readLines("models/rf_pred.csv")

raw_text <- paste(raw[-1], collapse=" ")
values <- as.numeric(unlist(strsplit(raw_text,"\\s+")))
values <- values[!is.na(values)]

# =========================
# MATCH LENGTH
# =========================
min_len <- min(length(values), nrow(data))
actual <- data$msl[1:min_len]
predicted <- values[1:min_len]

# =========================
# SCALE BOTH
# =========================
actual <- as.numeric(scale(actual))
predicted <- as.numeric(scale(predicted))

# =========================
# GRAPH 4: ACTUAL VS PREDICTED
# =========================
png("outputs/plots/actual_vs_predicted.png",800,600)

plot(actual, predicted,
     col="blue", pch=16,
     main="Actual vs Predicted")

abline(0,1,col="red")

dev.off()

# =========================
# GRAPH 5: RESIDUAL
# =========================
res <- actual - predicted

png("outputs/plots/residual_plot.png",800,600)

plot(predicted, res,
     col="purple", pch=16,
     main="Residual Plot")

abline(h=0,col="red")

dev.off()
# GRAPH 6:Model comparison
png("outputs/plots/model_comparison.png", 800, 600)

barplot(metrics$RMSE,
        names.arg = metrics$Model,
        col = "skyblue",
        main = "Model Comparison (RMSE)",
        ylab = "RMSE")

dev.off()
# GRAPH 7:FEATURE IMPORTANCE
library(caret)

# get importance
imp <- varImp(model)

png("outputs/plots/feature_importance.png", 800, 600)

plot(imp,
     main = "Feature Importance")

dev.off()