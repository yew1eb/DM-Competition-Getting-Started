#Benchmark code for last-man-standing hackathon by AnalyticsVidhya
#Code created on 29th Jan 2015
#Author: Bargava
#Link to competition
#http://datahack.analyticsvidhya.com/contest/last-man-standing

library(readr)
library(xgboost)

#Set the working directory. 
setwd("~/Documents/Kaggle/AnalyticsVidhya/last_man_standing/")
#Read the train, test and sample submission datasets
train <- read_csv("data/Train_Fyxd0t8.csv")
test <- read_csv("data/Test_C1XBIYq.csv")
#samplesub <- read_csv("~/Downloads/Sample_Submission_Psj3sjG.csv")

#samplesub's first column is test's ID column
samplesub <- as.matrix(test$ID)
names(samplesub) <- c("ID")

#Find the column types for train and test
lapply(train, class)
lapply(test, class)

#Certain columns are categorical. 
#One way to handle categorical is to one-hot encode them.

#the categorical variables are (refer to the data dictionary for this)
#Crop_Type : already a 0/1 variable. No need to change
#Soil_Type : already a 0/1 variable. No need to change
#Pesticide_Use_Category : Need to do one-hot encoding
#Season : Need to do one-hot encoding

#Combine train and test dataset
combined <- data.frame(rbind(train[, -ncol(train)], test))
#Drop the ID column
combined <- combined[,-1]
#Create a separate label data
label <- data.frame(label=train[, ncol(train)])

#Find frequencies of the labl
table(label)
#    0     1     2 
# 74238 12307  2313

#Create a new dataframe that has the data of those columns
#that are to be one-hot encoded
columns_for_one_hot <- c("Pesticide_Use_Category", "Season")
data_for_one_hot <- combined[, columns_for_one_hot]

#Remove those columns from the combined dataframe
combined <- combined[, !(names(combined) %in% columns_for_one_hot)]

for(i in 1:ncol(data_for_one_hot)){
  data_for_one_hot[,i] <- as.factor(data_for_one_hot[,i])
}

#check: class
lapply(data_for_one_hot, class)
#check:unique values
for(i in 1:ncol(data_for_one_hot)){
  print(names(data_for_one_hot)[i])
  print(unique(data_for_one_hot[,i]))
}

#one-hot encoding
one_hot_encoded_data <- matrix(nrow=nrow(data_for_one_hot))

for(i in 1:ncol(data_for_one_hot)){
  one_hot_encoded_data <- cbind(one_hot_encoded_data, 
                                model.matrix(~data_for_one_hot[,i]-1))
}

#Remove the first NA column
one_hot_encoded_data <- one_hot_encoded_data[, -1]

#Add the one-hot encoded columns to combined dataset
combined <- data.frame(cbind(combined, one_hot_encoded_data))

#Update train and test dataset
trainRows <- nrow(train)
testRows <- nrow(test)

train <- combined[1:trainRows,]
test <- combined[trainRows+1:testRows,]

#remove unnecessary dataframes
rm(combined, data_for_one_hot, one_hot_encoded_data, columns_for_one_hot, i, testRows, trainRows)

#Replace missing values 
#Find column names of train and test that has missing values
colnames(train)[colSums(is.na(train))>0]
colnames(test)[colSums(is.na(test))>0]

#Replace missing values with median
train <- apply(train, 2, function(x) x <- replace(x, is.na(x), median(x, na.rm=T)))
test <- apply(test, 2, function(x) x <- replace(x, is.na(x), median(x, na.rm=T)))


######################################
#xgboost 
######################################

dtrain <- xgb.DMatrix(data=train, label=as.matrix(label[,1]))

watchlist=list(train=dtrain)

model_xgb_1 <- xgboost(data=dtrain,
                    max.depth=4,
                    eta=0.3,
                    gamma=3,
                    min_child_weight=10,
                    subsample=0.5,
                    colsample_bytree=0.5,
                    base_score=0,
                    nround=120,
                    nthread=6,
                    objective="multi:softmax",
                    num_class=3,
                    verbose=2,
                    watchlist=watchlist,
                    eval.metric="merror",
                    set.seed=13
)

pred <- data.frame(predict(model_xgb_1, as.matrix(test)))
table(pred)
######################################
#End of xgboost model
######################################

#create sample submission
samplesub <- data.frame(cbind(samplesub, pred))
names(samplesub) <- c("ID", "Crop_Damage")
write.csv(samplesub, "submission/submission_29jan_2.csv", row.names=F)
