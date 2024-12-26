install.packages("glmnet")

#Load libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(reshape2)
library(nnet) 
library(caret)
library(randomForest)
library(ROCR)
library(ROSE)
library(neuralnet)
library(xgboost)
library(Matrix)
library(pROC)

#Load datasets
data <- read.csv("A1_data.csv")
str(data)
summary(data)
head(data)

#change from 0 to 1 and from 1 to 0 in isFraud
data$isFraud <- ifelse(data$isFraud == 1, 0, 1)

# Replace missing values with "unknown" in the specified columns
cleaned_data <- data %>%
  mutate(
    P_emaildomain = ifelse(is.na(P_emaildomain) | P_emaildomain == "", 
                           "Unknown", P_emaildomain),
    R_emaildomain = ifelse(is.na(R_emaildomain) | R_emaildomain == "", 
                           "Unknown", R_emaildomain),
    card4 = ifelse(is.na(card4) | card4 == "", "Unknown", card4),
    card6 = ifelse(is.na(card6) | card6 == "", "Unknown", card6),
    id_30 = ifelse(is.na(id_30) | id_30 == "", "Unknown", id_30),
    id_31 = ifelse(is.na(id_31) | id_31 == "", "Unknown", id_31),
    DeviceType = ifelse(is.na(DeviceType) | DeviceType == "", 
                        "Unknown", DeviceType),
    DeviceInfo = ifelse(is.na(DeviceInfo) | DeviceInfo == "", 
                        "Unknown", DeviceInfo)
  )

#Remove
cleaned_data <- cleaned_data %>%
  select(-c(addr1, addr2, dist1, dist2, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, 
            D12, D13, D14, D15, M1, M2, M3, M5, M6, M7, M8, M9, id_03, id_04, id_07,
            id_08, id_09, id_10, id_13, id_14, id_18, id_21, id_22, id_24, id_34, id_25, 
            id_26, id_32))

str(cleaned_data)
colSums(is.na(cleaned_data))

#Filled with mean
edited_col <- c("card2","card3","card5","D1","V310","V311","V312","V313","V314","id_02","id_05",
                "id_06","id_11","id_17","id_19","id_20","id_35","id_36","id_37","id_38")

for (x in edited_col) {
  missed_data_mean <- mean(cleaned_data[[x]], na.rm = TRUE)
  cleaned_data[[x]][is.na(cleaned_data[[x]])] <- missed_data_mean
}

str(cleaned_data)
colSums(is.na(cleaned_data))

#EDA
str(cleaned_data)
summary(cleaned_data)
head(cleaned_data)

#histogram of isFraud
fraud_count <- table(cleaned_data$isFraud)
barplot(fraud_count, 
        col = "red", 
        border = "black", 
        main = "Distribution of Fraud vs Non-Fraud", 
        xlab = "Fraud Status", 
        ylab = "Count")

#histogram of TransactionAmt
hist(cleaned_data$TransactionAmt, 
     breaks = 50, 
     col = "yellow", 
     border = "black", 
     main = "Transaction Amount Distribution", 
     xlab = "Transaction Amount", 
     ylab = "Frequency")

# Filter the dataset for non-fraud case
non_fraud_data <- cleaned_data %>% filter(isFraud == 1)

# Filter the dataset for fraud case
fraud_data <- cleaned_data %>% filter(isFraud == 0)

# ProductCD in non-fraud case
ggplot(non_fraud_data, aes(x = ProductCD)) +
  geom_bar(fill = "blue",color="black", alpha = 0.8) +
  labs(title = "Distribution of Product Code in non-fraudulent transaction", 
       x = "Product Code", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# ProductCD in fraud case
ggplot(fraud_data, aes(x = ProductCD)) +
  geom_bar(fill = "blue",color="black", alpha = 0.8) +
  labs(title = "Distribution of Product Code in fraudulent transaction", 
       x = "Product Code", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# Card Type in non-fraud case
ggplot(non_fraud_data, aes(x = card4)) +
  geom_bar(fill = "white",color="black", alpha = 0.8) +
  labs(title = "Distribution of Card Type in non-fraudulent transaction", 
       x = "Card Type", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# Card Type in fraud case
ggplot(fraud_data, aes(x = card4)) +
  geom_bar(fill = "white",color="black", alpha = 0.8) +
  labs(title = "Distribution of Card Type in fraudulent transaction", 
       x = "Card Type", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# credit and debit in non-fraud case
ggplot(non_fraud_data, aes(x = card6)) +
  geom_bar(fill = "black",color="white", alpha = 0.8) +
  labs(title = "Distribution of Credit and Debit Card in non-fraudulent transaction", 
       x = "Credit or Debit", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# credit and debit in fraud case
ggplot(fraud_data, aes(x = card6)) +
  geom_bar(fill = "black",color="white", alpha = 0.8) +
  labs(title = "Distribution of Credit and Debit Card during fraudulent transaction", 
       x = "Credit or Debit", y = "Count") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))  

# Plot boxplot for TransactionAmt vs isFraud
ggplot(cleaned_data, aes(x = as.factor(isFraud), y = TransactionAmt)) + 
  geom_boxplot(fill = "red" , alpha = 0.5) +
  labs(title = "Boxplot of Transaction Amount by Fraud Status", 
       x = "Fraud Status", 
       y = "Transaction Amount") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5))

# Convert TransactionDT to POSIXct
cleaned_data$TransactionDT <- as.POSIXct(cleaned_data$TransactionDT, 
                                         origin = "1970-01-01", tz = "UTC")

# Extract hour from TransactionDT
cleaned_data$hour <- format(cleaned_data$TransactionDT, "%H")

# Create a summary table to count occurrences of isFraud by hour
summary_data <- cleaned_data %>%
  group_by(hour, isFraud) %>%
  summarise(count = n(), .groups = 'drop')

# Plot the results
ggplot(summary_data, aes(x = hour, y = count, fill = as.factor(isFraud))) +
  geom_bar(stat = "identity" , position = "dodge") +
  labs(title = "Fraudulent and Non-Fraudulent Transactions by Hour",
       x = "Hour",
       y = "Number of Transactions",
       fill = "isFraud") +
  theme_minimal()

#Imbalanced data for training
imbalanced_data <- cleaned_data %>%
  select(isFraud, TransactionAmt, hour, ProductCD, card4, card6)
imbalanced_data = imbalanced_data %>% mutate_if(is.character, as.factor)
imbalanced_data$isFraud <- as.factor(cleaned_data$isFraud)
str(imbalanced_data)
set.seed(123)
table(imbalanced_data$isFraud)

#ROSE
rose_data <- ROSE(isFraud ~ ., data = imbalanced_data)$data
rose_data$isFraud <- factor(rose_data$isFraud, levels = c(0, 1))
table(rose_data$isFraud)

#logistic regression with imbalanced data
set.seed(123)
trainIndex <- createDataPartition(imbalanced_data$isFraud, p = 0.8, list = FALSE)
train_data <- imbalanced_data[trainIndex, ]
test_data <- imbalanced_data[-trainIndex, ]
table(train_data$isFraud)
table(test_data$isFraud)

log_reg_model <- glm(isFraud ~ ., data = train_data, family = binomial)
log_reg_pred <- predict(log_reg_model, newdata = test_data, type = "response")
log_reg_pred_class <- ifelse(log_reg_pred > 0.5, 1, 0)
test_data$isFraud <- factor(test_data$isFraud, levels = levels(test_data$isFraud))
conf_matrix_log <- confusionMatrix(as.factor(log_reg_pred_class), as.factor(test_data$isFraud))
print(conf_matrix_log)

roc_curve <- roc(test_data$isFraud, log_reg_pred)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression with imbalanced data")

#logistic regression with balanced data
set.seed(123)
trainIndex <- createDataPartition(rose_data$isFraud, p = 0.8, list = FALSE)
rose_train_data <- rose_data[trainIndex, ]
rose_test_data <- rose_data[-trainIndex, ]
table(rose_train_data$isFraud)
table(rose_test_data$isFraud)

log_reg_model <- glm(isFraud ~ ., data = rose_train_data, family = binomial)
log_reg_pred <- predict(log_reg_model, newdata = rose_test_data, type = "response")
log_reg_pred_class <- ifelse(log_reg_pred > 0.5, 1, 0)
log_reg_pred_class <- factor(log_reg_pred_class, levels = levels(rose_test_data$isFraud))
conf_matrix_log <- confusionMatrix(log_reg_pred_class, as.factor(rose_test_data$isFraud))
print(conf_matrix_log)

roc_curve <- roc(test_data$isFraud, log_reg_pred)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression with balanced data")

#Random forest with imbalanced data
set.seed(123) 
train_index <- sample(1:nrow(imbalanced_data), 0.8 * nrow(imbalanced_data))
train_data <- imbalanced_data[train_index, ]
test_data <- imbalanced_data[-train_index, ]

rf_model <- randomForest(isFraud ~ ., data = train_data, ntree = 500)

print(rf_model)

rf_pred <- predict(rf_model, test_data, type = "prob")[,2]
rf_pred_class <- ifelse(rf_pred > 0.5, 1, 0)
rf_pred_class <- factor(rf_pred_class, levels = levels(test_data$isFraud))
conf_matrix_rf <- confusionMatrix(as.factor(rf_pred_class), as.factor(test_data$isFraud))
print(conf_matrix_rf)

roc_curve <- roc(test_data$isFraud, rf_pred)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Random Forest with imbalanced data")

#Random forest with balanced data
set.seed(123) 
train_index <- sample(1:nrow(rose_data), 0.8 * nrow(rose_data))
rose_train_data <- rose_data[train_index, ]
rose_test_data <- rose_data[-train_index, ]

rf_model <- randomForest(isFraud ~ ., data = rose_train_data, ntree = 500)
rf_pred <- predict(rf_model, rose_test_data, type = "prob")[,2]
rf_pred_class <- ifelse(rf_pred > 0.5, 1, 0)
rf_pred_class <- factor(rf_pred_class, levels = levels(rose_test_data$isFraud))
conf_matrix_rf <- confusionMatrix(as.factor(rf_pred_class), as.factor(rose_test_data$isFraud))
print(conf_matrix_rf)

roc_curve <- roc(rose_test_data$isFraud, rf_pred)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Random Forest with balanced data")

#Neural Network with imbalanced data
trainIndex <- createDataPartition(imbalanced_data$isFraud, p = 0.8, list = FALSE)
train_data <- imbalanced_data[trainIndex, ]
test_data <- imbalanced_data[-trainIndex, ]
train_data$isFraud <- as.factor(train_data$isFraud)
test_data$isFraud <- as.factor(test_data$isFraud)

model <- nnet(
  isFraud ~ .,
  data = train_data,
  size = 5,  
  maxit = 200,
  linout = FALSE, 
  trace = TRUE
)
nn_prediction <- predict(model, test_data, type = "raw")
threshold <- 0.5
nn_pred_class <- factor(ifelse(nn_prediction > threshold, 1, 0), levels = c(0, 1))

conf_matrix <- confusionMatrix(nn_pred_class, test_data$isFraud)
print(conf_matrix)

roc_curve <- roc(test_data$isFraud, nn_prediction)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

plot(roc_curve, col = "blue", main = "ROC Curve for Neural Network with imbalanced data")

#Neural Network with balanced data
trainIndex <- createDataPartition(rose_data$isFraud, p = 0.8, list = FALSE)
rose_train_data <- rose_data[trainIndex, ]
rose_test_data <- rose_data[-trainIndex, ]

model <- nnet(
  isFraud ~ .,
  data = rose_train_data,
  size = 5,  
  maxit = 200,
  linout = FALSE, 
  trace = TRUE
)

nn_prediction <- predict(model, rose_test_data, type = "raw")
threshold <- 0.5
nn_pred_class <- factor(ifelse(nn_prediction > threshold, 1, 0), levels = c(0, 1))

conf_matrix <- confusionMatrix(nn_pred_class, rose_test_data$isFraud)
print(conf_matrix)

roc_curve <- roc(rose_test_data$isFraud, nn_prediction)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
plot(roc_curve, col = "blue", main = "ROC Curve for Neural Network with balanced data")

#More optimization
#Random forest number of trees adjustment
set.seed(123) 
train_index <- sample(1:nrow(rose_data), 0.8 * nrow(rose_data))
rose_train_data <- rose_data[train_index, ]
rose_test_data <- rose_data[-train_index, ]

rf_model <- randomForest(isFraud ~ ., 
                         data = rose_train_data, 
                         ntree = 500,
                         mtry = 3)

rf_pred <- predict(rf_model, rose_test_data, type = "prob")[,2]
rf_pred_class <- ifelse(rf_pred > 0.5, 1, 0)
rf_pred_class <- factor(rf_pred_class, levels = levels(rose_test_data$isFraud))
conf_matrix_rf <- confusionMatrix(as.factor(rf_pred_class), as.factor(rose_test_data$isFraud))
print(conf_matrix_rf)

roc_curve <- roc(rose_test_data$isFraud, rf_pred)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
plot(roc_curve, col = "blue", lwd = 2, 
     main = "ROC Curve for optimizaed Random Forest with balanced data")

#Neural Network with some adjustment
trainIndex <- createDataPartition(rose_data$isFraud, p = 0.8, list = FALSE)
rose_train_data <- rose_data[trainIndex, ]
rose_test_data <- rose_data[-trainIndex, ]

model <- nnet(
  isFraud ~ .,
  data = rose_train_data,
  size = 10,  
  maxit = 200,
  decay = 0.001,
  linout = FALSE, 
  trace = TRUE
)

nn_prediction <- predict(model, rose_test_data, type = "raw")
threshold <- 0.5
nn_pred_class <- factor(ifelse(nn_prediction > threshold, 1, 0), levels = c(0, 1))

conf_matrix <- confusionMatrix(nn_pred_class, rose_test_data$isFraud)
print(conf_matrix)

roc_curve <- roc(rose_test_data$isFraud, nn_prediction)
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))
plot(roc_curve, col = "blue", main = "ROC Curve for optimized Neural Network with balanced data")

