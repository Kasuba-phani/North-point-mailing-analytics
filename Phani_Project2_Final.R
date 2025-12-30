library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(Metrics)
library(randomForest)
library(glmnet)

software <- read.csv("Software_Mailing_List.csv")
summary(software)

#Data Preparation (creating 1 source column and removing sequence number)
source_cols <- grep("^source_", names(software), value = TRUE)

# Create a single variable that stores the source name where value = 1
software$source_combined <- apply(software[source_cols], 1, function(x) {
  sources <- source_cols[which(x == 1)]
  if(length(sources) == 0) return("None")
  paste(sources, collapse = ",")
})

# Identify all source columns
source_cols <- grep("^source_", names(software), value = TRUE)

software_clean <- software[, !(names(software) %in% c(source_cols, "sequence_number"))]
software_clean$source_combined <- software$source_combined

summary(software_clean)

#Converting the variables into categorical
software_clean$US <- as.factor(software_clean$US)
software_clean$Web.order <- as.factor(software_clean$Web.order)
software_clean$Gender.male <- as.factor(software_clean$Gender.male)
software_clean$Address_is_res <- as.factor(software_clean$Address_is_res)
software_clean$Purchase <- as.factor(software_clean$Purchase)

# Convert source_combined to factor
software_clean$source_combined <- as.factor(software_clean$source_combined)
summary(software_clean)

#Missing Values
sum(is.na(software_clean))

#Checking the Zeros
zero_counts <- colSums(software_clean == 0, na.rm = TRUE)
zero_counts

zero_df <- data.frame(
  Variable = names(zero_counts),
  Zeros = as.numeric(zero_counts)
)

library(ggplot2)
ggplot(zero_df, aes(x = reorder(Variable, Zeros), y = Zeros)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Count of Zero Values per Variable",
    x = "Variable",
    y = "Number of Zeros"
  ) +
  theme_minimal()

#Statistics and Diagrams
ggplot(software_clean, aes(x = US)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of US") +
  theme_minimal()

ggplot(software_clean, aes(x = Freq)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of Freq") +
  theme_minimal()

ggplot(software_clean, aes(x = last_update_days_ago)) +
  geom_histogram(binwidth = 200, fill = "steelblue") +
  ggtitle("Plot of last_update_days_ago") +
  theme_minimal()

ggplot(software_clean, aes(x = X1st_update_days_ago)) +
  geom_histogram(binwidth = 200, fill = "steelblue") +
  ggtitle("Plot of X1st_update_days_ago") +
  theme_minimal()

ggplot(software_clean, aes(x = Web.order)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of Web.order") +
  theme_minimal()

ggplot(software_clean, aes(x = Gender.male)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of Gender.male") +
  theme_minimal()

ggplot(software_clean, aes(x = Address_is_res)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of Address_is_res") +
  theme_minimal()

ggplot(software_clean, aes(x = Purchase)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Spending)) +
  geom_histogram(binwidth = 50, fill = "steelblue") +
  ggtitle("Plot of Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = source_combined)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Plot of source_combined") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Predictor Relevancy
ggplot(software_clean, aes(x = US, fill = Purchase)) +
  geom_bar(position = "dodge") +
  ggtitle("US vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Freq, fill = Purchase)) +
  geom_bar(position = "dodge") +
  ggtitle("Freq vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Purchase, y = last_update_days_ago, fill = Purchase)) +
  geom_boxplot() +
  ggtitle("last_update_days_ago vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Purchase, y = X1st_update_days_ago, fill = Purchase)) +
  geom_boxplot() +
  ggtitle("X1st_update_days_ago vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Web.order, fill = Purchase)) +
  geom_bar(position = "dodge") +
  ggtitle("Web.order vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Gender.male, fill = Purchase)) +
  geom_bar(position = "dodge") +
  ggtitle("Gender.male vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Address_is_res, fill = Purchase)) +
  geom_bar(position = "dodge") +
  ggtitle("Address_is_res vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = Purchase, y = Spending, fill = Purchase)) +
  geom_boxplot() +
  ggtitle("Spending vs Purchase") +
  theme_minimal()

ggplot(software_clean, aes(x = source_combined, fill = Purchase)) +
  geom_bar(position = "fill") +
  ggtitle("source_combined vs Purchase") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


ggplot(software_clean, aes(x = US, y = Spending, fill = US)) +
  geom_boxplot() +
  ggtitle("US vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = Freq, y = Spending, fill = Freq)) +
  geom_boxplot() +
  ggtitle("Freq vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = last_update_days_ago, y = Spending)) +
  geom_point(alpha = 0.4) +
  ggtitle("last_update_days_ago vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = X1st_update_days_ago, y = Spending)) +
  geom_point(alpha = 0.4) +
  ggtitle("X1st_update_days_ago vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = Web.order, y = Spending, fill = Web.order)) +
  geom_boxplot() +
  ggtitle("Web.order vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = Gender.male, y = Spending, fill = Gender.male)) +
  geom_boxplot() +
  ggtitle("Gender.male vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = Address_is_res, y = Spending, fill = Address_is_res)) +
  geom_boxplot() +
  ggtitle("Address_is_res vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = Purchase, y = Spending, fill = Purchase)) +
  geom_boxplot() +
  ggtitle("Purchase vs Spending") +
  theme_minimal()

ggplot(software_clean, aes(x = source_combined, y = Spending, fill = source_combined)) +
  geom_boxplot() +
  ggtitle("source_combined vs Spending") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#K-Means Clustering
software_numeric <- software_clean[, sapply(software_clean, is.numeric)]
software_scaled <- scale(software_numeric)

wss <- numeric(10)

for (i in 1:10) {
  wss[i] <- sum(kmeans(software_scaled, centers = i)$withinss)
}

plot(1:10, wss, type = "b",
     xlab = "Number of Clusters",
     ylab = "Within-Cluster Sum of Squares",
     main = "Elbow Method for Choosing K")

set.seed(123)
k3 <- kmeans(software_scaled, centers = 3, nstart = 25)
software_clean$cluster <- as.factor(k3$cluster)

library(ggplot2)

ggplot(software_clean, aes(x = last_update_days_ago,
                           y = X1st_update_days_ago,
                           color = cluster)) +
  geom_point(alpha = 0.6) +
  ggtitle("K-means Clustering Results") +
  theme_minimal()

software_clean$cluster <- as.factor(k3$cluster)

ggplot(software_clean, aes(x = cluster)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Cluster Sizes (K-means)") +
  xlab("Cluster") +
  ylab("Count") +
  theme_minimal()

#Classification
# Remove Spending from predictors
software_class <- software_clean[, !names(software_clean) %in% "Spending"]

# Train-test split (70/30)
set.seed(123)
train_index <- sample(1:nrow(software_class), 0.7 * nrow(software_class))

train <- software_class[train_index, ]
test  <- software_class[-train_index, ]

# Make sure Purchase is factor
train$Purchase <- as.factor(train$Purchase)
test$Purchase  <- as.factor(test$Purchase)

# Set positive class = "1"
test$Purchase <- relevel(test$Purchase, ref = "1")

#logistic Regression
# Logistic Regression model
log_model <- glm(Purchase ~ ., data = train, family = binomial)

# Prediction (probabilities)
log_prob <- predict(log_model, test, type = "response")

# Convert to class labels
log_pred <- ifelse(log_prob > 0.5, "1", "0")
log_pred <- factor(log_pred, levels = c("1", "0"))

# Confusion matrix
library(caret)
confusionMatrix(log_pred, test$Purchase, positive = "1")


#Naive Bayes
library(e1071)
library(e1071)

# Train Naive Bayes
nb_model <- naiveBayes(Purchase ~ ., data = train)

# Predict
nb_pred <- predict(nb_model, test)

# Ensure factor levels match and set positive = "1"
nb_pred <- factor(nb_pred, levels = c("1", "0"))
test$Purchase <- factor(test$Purchase, levels = c("1", "0"))

# Confusion Matrix (positive = 1)
library(caret)
confusionMatrix(nb_pred, test$Purchase, positive = "1")


#Decision Tree
library(rpart)
library(rpart.plot)

# Train the model
tree_model <- rpart(Purchase ~ ., data = train, method = "class")

# Predict class
tree_pred <- predict(tree_model, test, type = "class")

# Fix factor levels
tree_pred <- factor(tree_pred, levels = c("1", "0"))
test$Purchase <- factor(test$Purchase, levels = c("1", "0"))

# Confusion matrix with positive = 1
library(caret)
confusionMatrix(tree_pred, test$Purchase, positive = "1")

# Random Forest
library(randomForest)

# Train RF
rf_model <- randomForest(Purchase ~ ., data = train, ntree = 500)

# Predict
rf_pred <- predict(rf_model, test)

# Fix factor levels
rf_pred <- factor(rf_pred, levels = c("1", "0"))
test$Purchase <- factor(test$Purchase, levels = c("1", "0"))

# Confusion Matrix
library(caret)
confusionMatrix(rf_pred, test$Purchase, positive = "1")

#Regression
reg_data <- subset(software_clean, Purchase == "1")

# Remove Purchase (not needed for regression)
reg_data$Purchase <- NULL
set.seed(123)
train_idx <- sample(1:nrow(reg_data), 0.7 * nrow(reg_data))

#linear Regression
train_reg <- reg_data[train_idx, ]
test_reg  <- reg_data[-train_idx, ]

lin_model <- lm(Spending ~ ., data = train_reg)

lin_pred <- predict(lin_model, test_reg)

# Performance
lin_mse <- mean((lin_pred - test_reg$Spending)^2)
lin_mse

#Decision Tree Regression
library(rpart)
library(rpart.plot)

tree_reg <- rpart(Spending ~ ., data = train_reg, method = "anova")

tree_pred <- predict(tree_reg, test_reg)

# Performance
tree_mse <- mean((tree_pred - test_reg$Spending)^2)
tree_mse

# Plot tree
rpart.plot(tree_reg)

#Random Forest Regression
library(randomForest)

rf_reg <- randomForest(Spending ~ ., data = train_reg, ntree = 500)

rf_pred <- predict(rf_reg, test_reg)

# Performance
rf_mse <- mean((rf_pred - test_reg$Spending)^2)
rf_mse

# Variable Importance Plot
varImpPlot(rf_reg)

# STEPWISE REGRESSION (using AIC)

# Make sure MASS is loaded
library(MASS)

# Full model with all predictors
full_model <- lm(Spending ~ ., data = train_reg)

# Null model (intercept only)
null_model <- lm(Spending ~ 1, data = train_reg)

# Stepwise selection (both directions by default)
step_model <- stepAIC(
  object = null_model,                       
  scope  = list(lower = null_model, 
                upper  = full_model),     
  direction = "both",                   
  trace = TRUE                            
)

# View selected model
summary(step_model)

# Predict on test set
step_pred <- predict(step_model, newdata = test_reg)

# MSE for stepwise model
step_mse <- mean((step_pred - test_reg$Spending)^2)
step_mse

