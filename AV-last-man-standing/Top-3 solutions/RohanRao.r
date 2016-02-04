## setting working directory
path <- "./LastManStanding"
setwd(path)


## loading libraries
library(data.table)
library(plyr)
library(xgboost)


## loading data
train <- read.csv("./Train_Fyxd0t8.csv")
test <- read.csv("./Test_C1XBIYq.csv")


## preparing data
X_train <- train
X_test <- test

# converting missing values to -1
X_train[is.na(X_train)] <- -1
X_test[is.na(X_test)] <- -1

# removing rows (considering them as outliers) where Number_Doses_Week = 0 (improves score by ~ 0.02%)
X_train <- X_train[X_train$Number_Doses_Week > 0,]

# extracting ids and target
train_ids <- X_train$ID
test_ids <- X_test$ID

target <- X_train$Crop_Damage

# creating panel of train and test
X_panel <- rbind.fill(X_train, X_test)

# converting ID to numeric and ordering by ID
X_panel$ID <- as.integer(substr(X_panel$ID,2,9))
X_panel <- X_panel[order(X_panel$ID),]

## feature engineering

# Group: Group/Cluster based on increasing sequence of Estimated_Insects_Count and Number_Doses_Week
# Group_Change: Binary value indicating if the observation is the first observation of a new group in the same batch
# Group_First: Binary value indicating if the observation is the first observation of the group
# Group_Last: Binary value indicating if the observation is the last observation of the group
# Group_Avg: If the nearest observation before and after has the same Crop Damage in train data, then that Crop Damage
# Group_One: Binary value indicating whether an observation having Crop Damage 1 was seen previously in group
# Group_Two: Binary value indicating whether an observation having Crop Damage 2 was seen previously in group
# Group_Count: Total number of observations in the group
# Group_SD: Standard Deviation of Number_Doses_Week values in the group
# Batch_Last: Binary value indicating if the observation is the last observation of the batch
# Soil_Change: Binary value for last row of group indicating if the Soil Type of the next observation is different

# calculating Group, Group_Change, Group_First, Group_Last, Batch_Last
X_panel$Group <- 0
X_panel$Group[1] <- 1

X_panel$Group_Change <- 0
X_panel$Group_First <- 0
X_panel$Group_Last <- 0
X_panel$Batch_Last <- 0

X_panel$Group_First[1] <- 1

X_panel$Soil_Change <- 0

for (i in 2:nrow(X_panel))
{
  if (abs(X_panel$Estimated_Insects_Count[i] - X_panel$Estimated_Insects_Count[i-1]) > 1)
  {
    X_panel$Group[i] <- X_panel$Group[i-1] + 1
    X_panel$Group_First[i] <- 1
    X_panel$Group_Last[i-1] <- 1
    X_panel$Batch_Last[i-1] <- 1
    X_panel$Group_Num <- 1
  }else
  {
    if (X_panel$Number_Doses_Week[i] >= X_panel$Number_Doses_Week[i-1])
    {
      if (X_panel$Soil_Type[i] == X_panel$Soil_Type[i-1])
      {
        X_panel$Group[i] <- X_panel$Group[i-1]
        X_panel$Group_Num[i] <- X_panel$Group_Num[i-1] + 1
      }else
      {
        X_panel$Group[i] <- X_panel$Group[i-1] + 1
        X_panel$Group_Num <- 1
      }
    }else
    {
      X_panel$Group[i] <- X_panel$Group[i-1] + 1
      X_panel$Group_Change[i] <- 1
      X_panel$Group_First[i] <- 1
      X_panel$Group_Last[i-1] <- 1
      X_panel$Group[i] <- X_panel$Group[i-1] + 1
    }
  }
  
  if (i %% 10000 == 0 | i == nrow(X_panel))
  {
    cat(i, "completed,", nrow(X_panel)-i, "remaining\n")
  }
}

# calculating Soil_Change for last observations
for (i in 1:(nrow(X_panel)-1))
{
  print(i)
  
  if (X_panel$Group_Last[i] == 1 & X_panel$Soil_Type[i] != X_panel$Soil_Type[i+1])
  {
    X_panel$Soil[i] <- 1
  }
}

# calculating Group_Avg for groups having more at least two observations in train data
group_count <- data.frame(table(X_panel$Group[!is.na(X_panel$Crop_Damage)]))
group_count <- subset(group_count, Freq > 1)
group_count$Var1 <- as.numeric(as.character(group_count$Var1))

X_avg <- data.frame()

ldf <- lapply(unique(group_count$Var1), function(k)
{
  print(k)
  
  X_group <- subset(X_panel, Group == k, select=c("ID","Crop_Damage"))
  
  if (is.unsorted(X_group$Crop_Damage, na.rm=T) == F)
  {
    for (i in unique(X_group$Crop_Damage[!is.na(X_group$Crop_Damage)]))
    {
      id_min <- min(X_group$ID[X_group$Crop_Damage == i], na.rm=T) + 1
      id_max <- max(X_group$ID[X_group$Crop_Damage == i], na.rm=T) - 1
      
      if (id_min <= id_max)
      {
        X_avg <<- rbind(X_avg, data.frame("ID"=seq(id_min, id_max),
                                          "Group_Avg"=i))
      }
    }
  }
})

# calculating Group_One, Group_Two for all groups
X_one <- data.frame()
X_two <- data.frame()

ldf <- lapply(unique(X_panel$Group), function(k)
{
  print(k)
  
  X_group <- subset(X_panel, Group == k)
  
  if (1 %in% unique(X_group$Crop_Damage))
  {
    id_min_1 <- min(X_group$ID[X_group$Crop_Damage == 1], na.rm=T) + 1
    id_max <- max(X_group$ID)
    
    if (id_min_1 <= id_max)
    {
      X_one <<- rbind(X_one, data.frame("ID"=seq(id_min_1, id_max),
                                        "Group_One"=1))
    }
  }
  
  if (2 %in% unique(X_group$Crop_Damage))
  {
    id_min_2 <- min(X_group$ID[X_group$Crop_Damage == 2], na.rm=T) + 1
    id_max <- max(X_group$ID)
    
    if (id_min_2 <= id_max)
    {
      X_two <<- rbind(X_two, data.frame("ID"=seq(id_min_2, id_max),
                                        "Group_Two"=1))
    }
  }  
})

# merging features with data
X_panel <- merge(X_panel, X_avg, all.x=T, by="ID")
X_panel$Group_Avg[is.na(X_panel$Group_Avg)] <- -1

X_panel <- merge(X_panel, X_one, all.x=T, by="ID")
X_panel$Group_One[is.na(X_panel$Group_One)] <- 0

X_panel <- merge(X_panel, X_two, all.x=T, by="ID")
X_panel$Group_Two[is.na(X_panel$Group_Two)] <- 0

# calculating Group_Count, Group_SD
X_panel <- data.table(X_panel)

X_group <- X_panel[, .(Group_Count = .N,
                       Group_SD = sd(Number_Doses_Week)), .(Group)]

X_group[is.na(X_group)] <- -1

X_panel <- merge(X_panel, X_group, by="Group")

# creating train and test dataframes
X_train <- subset(X_panel, !is.na(Crop_Damage), select=-c(ID, Crop_Damage, Group))
X_test <- subset(X_panel, is.na(Crop_Damage), select=-c(ID, Crop_Damage, Group))


## XGBoost

# cross-validation
set.seed(23)
model_xgb <- xgb.cv(data=as.matrix(X_train), label=as.matrix(target), objective="multi:softprob", num_class=3, nfold=5, nrounds=130, eta=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, min_child_weight=1, eval_metric="merror", prediction=T)

# CV Error: 0.039207
# CV Accuracy = 1 - Error = 96.08%
# LB Score: 96.04%

# model building
set.seed(23)
model_xgb <- xgboost(as.matrix(X_train), as.matrix(target), objective="multi:softprob", num_class=3, nrounds=130, eta=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, min_child_weight=1, eval_metric='merror')

# variable importance
print(xgb.importance(feature_names=colnames(X_train), model=model_xgb))

#                  Feature      Gain       Cover     Frequence
#1:             Group_Count 0.460615421 0.203682099 0.09762933
#2:               Group_One 0.142286750 0.113337814 0.03226539
#3:            Group_Change 0.075315590 0.042351033 0.02332803
#4: Estimated_Insects_Count 0.074563744 0.105871825 0.18624555
#5:               Group_Avg 0.074135191 0.106491231 0.02832690
#6:             Soil_Change 0.025121602 0.045855656 0.02885708
#7:       Number_Weeks_Quit 0.020677026 0.040302551 0.09596304
#8:             Group_First 0.018942027 0.016997088 0.01166402
#9:              Batch_Last 0.017457599 0.049503374 0.03143225
#10:       Number_Doses_Week 0.016825784 0.038309803 0.07558888
#11:                Group_SD 0.016379250 0.072851704 0.13277285
#12:       Number_Weeks_Used 0.011084116 0.040653870 0.11959403
#13:              Group_Last 0.010910592 0.042932419 0.02416118
#14:               Soil_Type 0.010022804 0.013245355 0.02688783
#15:               Group_Two 0.008371727 0.038155935 0.01363327
#16:               Crop_Type 0.008151269 0.017183499 0.03120503
#17:  Pesticide_Use_Category 0.007646653 0.008463722 0.01302734
#18:                  Season 0.001492855 0.003811024 0.02741801

# scoring
pred <- predict(model_xgb, as.matrix(X_test))
pred_matrix <- as.data.frame(matrix(pred, nrow(X_test), 3, byrow=T))

# creating submission file
submit <- as.data.frame(apply(pred_matrix, 1, function(y) order(-y)[1:1]))
names(submit) <- "Crop_Damage"
submit$Crop_Damage <- submit$Crop_Damage - 1
submit$ID <- test_ids

# predicting outliers observations as 1
submit$Crop_Damage[submit$ID %in% test$ID[test$Number_Doses_Week == 0]] <- 1

# saving output
submit <- submit[,c("ID","Crop_Damage")]
write.csv(submit, "./submit.csv", row.names=F)
