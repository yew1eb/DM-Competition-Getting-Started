library(dplyr)   # for data frame creation and manipulation
library(corrplot)
library(GGally) # for ggpairs

# -------------------------------------------------------------------------------------------
## Load the data 
train <- read.csv("data/Training.csv")
test <- read.csv("data/Testing.csv")

dim(train)
dim(test)

head(train)
str(train)
summary(train)

sum(is.na(train))  # check missing values
nrow(train) - nrow(distinct(train))      # check for duplicate rows

colnames(train) <- c("Id","MonthsLast","NumberDonations","Volume","MonthsFirst","Donated2007")
colnames(test) <- c("Id","MonthsLast","NumberDonations","Volume","MonthsFirst")

labels <- train$Donated2007
feats_train <- train[,c(2:5)]
feats_test <- test[,c(2:5)]

feats_train$Consistency <- (feats_train$MonthsFirst - feats_train$MonthsLast)/feats_train$NumberDonations
feats_train$Consistency[feats_train$Consistency == 0] <- feats_train$MonthsLast[feats_train$Consistency == 0]

feats_test$Consistency <- (feats_test$MonthsFirst - feats_test$MonthsLast)/feats_test$NumberDonations
feats_test$Consistency[feats_test$Consistency == 0] <- feats_test$MonthsLast[feats_test$Consistency == 0]

## -----------------------------------------------------------------------------------
# correlation matrix
train <- data.frame(feats_train,labels)
colnames(train) <- c("MonthsLast","NumberDonations","Volume","MonthsFirst","Consistency","Donated2007")

M <- cor(train)

windows()
corrplot(M,method="circle")

## ----------------------------------------------------------------------------------
# pairwise plots
windows()
train$Donated2007 <- as.factor(train$Donated2007)
ggpairs(train,colour = "Donated2007")


## ----------------------------------------------------------------------------------
# End of exploratory analysis

## ----------------------------------------------------------------------------------
# See code for exploratory data analysis in previous post

library(caret)
library(e1071) # for confusion matrix
library(rattle) # for fancyRpartPlot

## ----------------------------------------------------------------------------------
# Pre-processing
# procValues <- preProcess(feats_train, method = c("center", "scale", "YeoJohnson"))
# feats_train <- predict(procValues, feats_train)

feats_train$Volume <- NULL
feats_test$Volume <- NULL

## ----------------------------------------------------------------------------------
# Split data
labels[labels == 0] <- c("No")
labels[labels == 1] <- c("Yes")

set.seed(1)
inTrainingSet <- createDataPartition(labels, p = .75, list = FALSE)
CVTrainFeats <- feats_train[inTrainingSet,]
CVTestFeats <- feats_train[-inTrainingSet,]
CVTrainLabels <- as.factor(labels[inTrainingSet])
CVTestLabels <- as.factor(labels[-inTrainingSet])


# -------------------------------------------------------------------------------------------
# 1. Linear Discriminant Analysis
LDA_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "lda", metric = "logLoss", trControl = ctrl)
LDA_Model$results
LDA_Pred_Train <- predict(LDA_Model, CVTrainFeats)
LDA_Pred_Test <- predict(LDA_Model, CVTestFeats)
LDA_Probs_Train <- predict(LDA_Model, CVTrainFeats, type = "prob")
LDA_Probs_Test <- predict(LDA_Model, CVTestFeats, type = "prob")
confusionMatrix(LDA_Pred_Test, CVTestLabels,positive="Yes")

LDA_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = LDA_Pred_Train, LDA_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
LDA_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = LDA_Pred_Test, LDA_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

LDA_featureImportance <- varImp(LDA_Model)

windows()
plot(LDA_featureImportance)

# -------------------------------------------------------------------------------------------
# 2. Logistic Regression
GLM_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "glm", metric = "logLoss", trControl = ctrl)
GLM_Model$results
GLM_Pred_Train <- predict(GLM_Model, CVTrainFeats)
GLM_Pred_Test <- predict(GLM_Model, CVTestFeats)
GLM_Probs_Train <- predict(GLM_Model, CVTrainFeats, type = "prob")
GLM_Probs_Test <- predict(GLM_Model, CVTestFeats, type = "prob")
confusionMatrix(GLM_Pred_Test, CVTestLabels,positive="Yes")

GLM_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = GLM_Pred_Train, GLM_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
GLM_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = GLM_Pred_Test, GLM_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

GLM_featureImportance <- varImp(GLM_Model)
  
windows()
plot(GLM_featureImportance)

# -------------------------------------------------------------------------------------------
# 3. Naive Bayes
NB_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "nb", metric = "logLoss", trControl = ctrl)
NB_Model$results
NB_Pred_Train <- predict(NB_Model, CVTrainFeats)
NB_Pred_Test <- predict(NB_Model, CVTestFeats)
NB_Probs_Train <- predict(NB_Model, CVTrainFeats, type = "prob")
NB_Probs_Test <- predict(NB_Model, CVTestFeats, type = "prob")
confusionMatrix(NB_Pred_Test, CVTestLabels,positive="Yes")

NB_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = NB_Pred_Train, NB_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
NB_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = NB_Pred_Test, NB_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

NB_featureImportance <- varImp(NB_Model)

windows()
plot(NB_featureImportance)

# -------------------------------------------------------------------------------------------
# 4. Train CART
CART_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "rpart", metric = "logLoss", trControl = ctrl)
CART_Model$results
CART_Pred_Train <- predict(CART_Model, CVTrainFeats)
CART_Pred_Test <- predict(CART_Model, CVTestFeats)
CART_Probs_Train <- predict(CART_Model, CVTrainFeats, type = "prob")
CART_Probs_Test <- predict(CART_Model, CVTestFeats, type = "prob")
confusionMatrix(CART_Pred_Test, CVTestLabels,positive="Yes")

CART_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = CART_Pred_Train, CART_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
CART_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = CART_Pred_Test, CART_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

CART_featureImportance <- varImp(CART_Model)


library(rpart.plot)
windows()
fancyRpartPlot(CART_Model$finalModel)

# -------------------------------------------------------------------------------------------
# 5. Train Support Vector Machine model
SVM_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "svmRadial", metric = "logLoss", trControl = ctrl)
SVM_Model$results
SVM_Pred_Train <- predict(SVM_Model, CVTrainFeats)
SVM_Pred_Test <- predict(SVM_Model, CVTestFeats)
SVM_Probs_Train <- predict(SVM_Model, CVTrainFeats, type = "prob")
SVM_Probs_Test <- predict(SVM_Model, CVTestFeats, type = "prob")
confusionMatrix(SVM_Pred_Test, CVTestLabels,positive="Yes")

SVM_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = SVM_Pred_Train, SVM_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
SVM_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = SVM_Pred_Test, SVM_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

SVM_featureImportance <- varImp(SVM_Model)

windows()
plot(SVM_featureImportance)

# -------------------------------------------------------------------------------------------
# 6. Random Forest
RF_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "rf", metric = "logLoss", trControl = ctrl)
RF_Model$results
RF_Pred_Train <- predict(RF_Model, CVTrainFeats)
RF_Pred_Test <- predict(RF_Model, CVTestFeats)
RF_Probs_Train <- predict(RF_Model, CVTrainFeats, type = "prob")
RF_Probs_Test <- predict(RF_Model, CVTestFeats, type = "prob")
confusionMatrix(RF_Pred_Test, CVTestLabels,positive="Yes")

RF_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = RF_Pred_Train, RF_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
RF_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = RF_Pred_Test, RF_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

RF_featureImportance <- varImp(RF_Model)

windows()
plot(RF_featureImportance)

# -------------------------------------------------------------------------------------------
# 7. Gradient Boosting Machine
GBM_Model <- train(x=CVTrainFeats, y=CVTrainLabels, method = "gbm", metric = "logLoss", trControl = ctrl,verbose=FALSE)
GBM_Model$results
GBM_Pred_Train <- predict(GBM_Model, CVTrainFeats)
GBM_Pred_Test <- predict(GBM_Model, CVTestFeats)
GBM_Probs_Train <- predict(GBM_Model, CVTrainFeats, type = "prob")
GBM_Probs_Test <- predict(GBM_Model, CVTestFeats, type = "prob")
confusionMatrix(GBM_Pred_Test, CVTestLabels,positive="Yes")

GBM_LogLoss_Train <- data.frame(obs = CVTrainLabels, pred = GBM_Pred_Train, GBM_Probs_Train) %>%
  mnLogLoss(lev=c("Yes","No"))
GBM_LogLoss_Test <- data.frame(obs = CVTestLabels, pred = GBM_Pred_Test, GBM_Probs_Test) %>%
  mnLogLoss(lev=c("Yes","No"))

GBM_featureImportance <- varImp(GBM_Model)

windows()
plot(GBM_featureImportance)

# -------------------------------------------------------------------------------------------
# Ensemble: mean probabilities
labels <- as.factor(labels)

ctrl <- trainControl(method = "cv",
                     p = 1,            # use full training set
                     repeats = 1,
                     classProbs = TRUE,
                     summaryFunction = mnLogLoss)

# Linear Discriminant Analysis
LDA_Model <- train(x=feats, y=labels, method = "lda", metric = "logLoss", trControl = ctrl)
LDA_Probs <- predict(LDA_Model, feats, type = "prob")

# Logistic Regression
GLM_Model <- train(x=feats, y=labels, method = "glm", metric = "logLoss", trControl = ctrl)
GLM_Probs <- predict(GLM_Model, feats, type = "prob")

# Naive Bayes
NB_Model <- train(x=feats, y=labels, method = "nb", metric = "logLoss", trControl = ctrl)
NB_Probs <- predict(NB_Model, feats, type = "prob")

# Classification tree
CART_Model <- train(x=feats, y=labels, method = "rpart", metric = "logLoss", trControl = ctrl)
CART_Probs <- predict(CART_Model, feats, type = "prob")

# Support Vector Machine
SVM_Model <- train(x=feats, y=labels, method = "svmRadial", metric = "logLoss", trControl = ctrl)
SVM_Probs <- predict(SVM_Model, feats, type = "prob")

# Random Forest
RF_Model <- train(x=feats, y=labels, method = "rf", metric = "logLoss", trControl = ctrl)
RF_Probs <- predict(RF_Model, feats, type = "prob")

# Gradient Boosting Machine
GBM_Model <- train(x=feats, y=labels, method = "gbm", metric = "logLoss", trControl = ctrl, verbose=FALSE)
GBM_Probs <- predict(GBM_Model, feats, type = "prob")

# Calculate means of class probabilities
X_train_0 <- data.frame(LDA_Probs$No,GLM_Probs$No,NB_Probs$No,CART_Probs$No,SVM_Probs$No,RF_Probs$No,GBM_Probs$No)
X_train_1 <- data.frame(LDA_Probs$Yes,GLM_Probs$Yes,NB_Probs$Yes,CART_Probs$Yes,SVM_Probs$Yes,RF_Probs$Yes,GBM_Probs$Yes)

# Ensemble LogLoss
X_train = data.frame("No" = apply(X_train_0,1,mean), "Yes" = apply(X_train_1,1,mean))
Ensemble_LogLoss <- data.frame(obs = labels, pred = ifelse(X_train$Yes >= X_train$No, "Yes", "No"), X_train) %>%
  mnLogLoss(lev=c("Yes","No"))

# -------------------------------------------------------------------------------------------
# Final CART for report
CART_Model <- train(x=feats, y=labels, method = "rpart", metric = "logLoss", trControl = ctrl)
CART_featureImportance <- varImp(CART_Model)


library(rpart.plot)
windows()
fancyRpartPlot(CART_Model$finalModel)

# -------------------------------------------------------------------------------------------

submission <- read.csv("data/submission.csv")
submission$Made.Donation.in.March.2007 <- 0.24
colnames(submission) <- c("","Made Donation in March 2007")
write.csv(submission,"submission_chance.csv",row.names=FALSE)

# -------------------------------------------------------------------------------------------

CART_Model <- train(x=feats_train, y=labels, method = "rpart", metric = "logLoss", trControl = ctrl)
CART_Model$results
CART_Probs_Test <- predict(CART_Model, feats_test, type = "prob")

submission <- read.csv("data/submission.csv")
submission$Made.Donation.in.March.2007 <- CART_Probs_Test$Yes
colnames(submission) <- c("","Made Donation in March 2007")
write.csv(submission,"submission_CART.csv",row.names=FALSE)

# -------------------------------------------------------------------------------------------

# Linear Discriminant Analysis
LDA_Model <- train(x=feats_train, y=labels, method = "lda", metric = "logLoss", trControl = ctrl)
LDA_Probs_Test <- predict(LDA_Model, feats_test, type = "prob")

# Logistic Regression
GLM_Model <- train(x=feats_train, y=labels, method = "glm", metric = "logLoss", trControl = ctrl)
GLM_Probs_Test <- predict(GLM_Model, feats_test, type = "prob")

# Naive Bayes
NB_Model <- train(x=feats_train, y=labels, method = "nb", metric = "logLoss", trControl = ctrl)
NB_Probs_Test <- predict(NB_Model, feats_test, type = "prob")

# Classification tree
CART_Model <- train(x=feats_train, y=labels, method = "rpart", metric = "logLoss", trControl = ctrl)
CART_Probs_Test <- predict(CART_Model, feats_test, type = "prob")

# Support Vector Machine
SVM_Model <- train(x=feats_train, y=labels, method = "svmRadial", metric = "logLoss", trControl = ctrl)
SVM_Probs_Test <- predict(SVM_Model, feats_test, type = "prob")

# Random Forest
RF_Model <- train(x=feats_train, y=labels, method = "rf", metric = "logLoss", trControl = ctrl)
RF_Probs_Test <- predict(RF_Model, feats_test, type = "prob")

# Gradient Boosting Machine
GBM_Model <- train(x=feats_train, y=labels, method = "gbm", metric = "logLoss", trControl = ctrl, verbose=FALSE)
GBM_Probs_Test <- predict(GBM_Model, feats_test, type = "prob")

# Prepare submission file
submission <- read.csv("data/submission.csv")
X_train_1 <- data.frame(LDA_Probs_Test$Yes,GLM_Probs_Test$Yes,NB_Probs_Test$Yes,CART_Probs_Test$Yes,SVM_Probs_Test$Yes,RF_Probs_Test$Yes,GBM_Probs_Test$Yes)
submission$Made.Donation.in.March.2007 <-  apply(X_train_1,1,mean)
colnames(submission) <- c("","Made Donation in March 2007")
write.csv(submission,"submission_ensemble.csv",row.names=FALSE)