#Data Pre Processing
#Data Pre Processing
#Data Pre Processing

require(readr)
require(dplyr)

options(dplyr.print_max = Inf)
options(dplyr.width = Inf)


setwd("~/AV/Last man standing")

train <- read_csv("./Train_Fyxd0t8.csv")
test <- read_csv("./Test_C1XBIYq.csv")
sample_submission <- read_csv("./Sample_Submission_Psj3sjG.csv")

train$trainFlag <- 1
test$trainFlag <- 0
test$Crop_Damage <- NA

alldata <- rbind(train, test)

alldata$Season_cropType <- as.integer(as.factor(paste(alldata$Season, alldata$Crop_Type, sep = "_")))
alldata$Season_soilType <- as.integer(as.factor(paste(alldata$Season, alldata$Soil_Type, sep = "_")))
alldata$cropType_soilType <- as.integer(as.factor(paste(alldata$Crop_Type, alldata$Soil_Type, sep = "_")))
alldata$insect_cnt_by_dose_freq <- alldata$Estimated_Insects_Count/alldata$Number_Doses_Week
alldata$total_dosage <- alldata$Number_Doses_Week * alldata$Number_Weeks_Used
alldata$weeks_since_pesticide <- alldata$Number_Weeks_Used + alldata$Number_Weeks_Quit
alldata$weeks_used_by_weeks_quit <- alldata$Number_Weeks_Used / alldata$Number_Weeks_Quit

save(alldata, file = "./alldata_LMS.RData")

#Automate Functions
#Automate Functions
#Automate Functions

# reshape the xgboost probabilities for multi:softprob
get_proba <- function(preds, Id = NULL, num_class = 3){
  
  p <- data.frame(t(matrix(preds, nrow = num_class, ncol = length(preds)/num_class)))
  cnames <- c("0", "1", "2")
  names(p) <- cnames
  
  if(is.null(Id)){
    
    return(p)
    
  } else {
    
    p <- cbind(id = Id, p)
    return(p)
    
  }
  
}

# hardcoded class value
get_predict <- function(predMatrix, isTest = FALSE) {
  
  if(ncol(predMatrix) > 3) {
    
    k <- apply(predMatrix[, -1], 1, which.max) - 1
    if(isTest) {
      df <- cbind.data.frame(ID = predMatrix[, 1], Crop_Damage = k)
    } else {
      
      df <- cbind.data.frame(ID = predMatrix[, 1], predicted_class = k)
    }
    return(df)
  }
  else {
    
    k <- apply(predMatrix[, -1], which.max) - 1
    return(k)
    
  }
  
}

# find time blocks of similar Estimated_Insects_Counts
# returns the relevant block and start point of the next block
# assumption: max block length = 100

findBlock <- function(i) {
  
  d <- c()
  for (j in 1:100) {
    
    k <- abs(md$Estimated_Insects_Count[i] - md$Estimated_Insects_Count[i + j])
    d <- c(d, k)
  }
  
  ct <- min(which(d > 1))
  st <- i
  en <- i + ct - 1
  ak <- md[i:en,]
  
  new_start <- i + ct
  
  p <- list(block = ak, ns = new_start)
  return(p)
}

# same function just returning the block this time
# i wrote it for achieving something else
# could be ignored
findBlock2 <- function(i) {
  
  d <- c()
  for (j in 1:100) {
    
    k <- abs(ss$Estimated_Insects_Count[i] - ss$Estimated_Insects_Count[i + j])
    d <- c(d, k)
  }
  
  ct <- min(which(d > 1))
  st <- i
  en <- i + ct - 1
  ak <- ss[i:en,]
  
  return(ak)
}

# find the test entries inside a block
findTest <- function(df) {
  
  non_na <- which(is.na(df$Crop_Damage))
  
  return(non_na)
  
}

# find the train entries inside the block
findTrain <- function(df) {
  
  non_na <- which(!is.na(df$Crop_Damage))
  
  return(non_na)
  
}

# in a block search for the previous and next non NA Crop_Damage
find_non_na <- function(df, i){
  
  if(i == 1){
    
    return(-1)
    
  }else{
    
    for(j in 1:(i-1)){
      
      if(!is.na(df$Crop_Damage[i - j])){
        
        p1 <- i - j
        break
        
      }
      
    }
    for(j in 1:(nrow(df) - i)){
      
      if(!is.na(df$Crop_Damage[i + j])){
        
        p2 <- i + j
        break
      }
      
    }
    
    p <- list(last = p1, nxt = p2)
    return(p)
  }
  
}

# if a blocks starts with NA i.e. if the starting point
# of a block is a test entry just impute with 0
# this is just an assumption on my part keeping an eye on
# the pattern present in the data

takeFirst <- function(block) {
  
  if(is.na(block$Crop_Damage[1])){
    
    block$Crop_Damage[1] <- 0
  }
  return(block)
}

# if a block ends with NA i.e. if the ending point of a
# block is a test entry just impute with the last present
# Crop_Damage value in the block 
# This is a more vulnerable assumption but strong enough
# to lift the score above 90%
# correction will be applied to this assumption going ahead

takeLast <- function(block) {
  
  if(is.na(block$Crop_Damage[nrow(block)])){
    
    last_present <- max(findTrain(block))
    block$Crop_Damage[nrow(block)] <- block$Crop_Damage[last_present]
    
  }
  return(block)
}

# impute the intermittent test entries in a block with the mean of
# previous and next train entries

impute_intermittent <- function(block, index){
  
  p <- find_prev_non_na(block, index)
  block$Crop_Damage[index] <- (block$Crop_Damage[p$last] + block$Crop_Damage[p$nxt])/2
  return(block)
}


# this function rectifies any error made due to the assumption of the takeLast
# function by observing the pattern in Numnber_Doses_Week values in the block
# this asks for manual user input interactively in the run time. I 
# put it in that way to verify my assumption of the pattern of Number_Doses_Week
# values in the data. This is automated once the pattern is confirmed.

end_of_block_correction <- function(block) {
  
  if(nrow(block) == 1){
    
    block$corrected <- block$Crop_Damage
    
  }else{
    
    cd <- unique(block$CD[!is.na(block$CD)])
    if(length(cd) == 1 & is.na(block$CD[nrow(block)])){
      
      if(cd == 0){
        
        k <- block$Number_Doses_Week
        k_diff <- diff(k)
        is_reverse <- sum(k_diff < 0)
        if(is_reverse == 0){
          
          block$corrected <- block$Crop_Damage
        }else{
          
          pt <- min(which(k_diff < 0))
          
          if(nrow(block) >= 5 | pt == 1){
            
            cat("\n----------------------------")
            cat("\nBlock starting ID: ", block$ID[1])
            cat("\nBlock length: ", nrow(block))
            cat("\nBlock Number_Doeses_Week: ", block$Number_Doses_Week)
            cat("\nReverse indices: ", which(k_diff < 0))
            
            n <- readline(prompt="\nEnter the index you want to use: ")
            n <- as.integer(n)
            if(n == -1){
              
              block$corrected <- block$Crop_Damage
            }else{
              block$corrected <- block$Crop_Damage
              block$corrected[(n+1):nrow(block)] <- 1  
              
            }
            
          }else {
            
            block$corrected <- block$Crop_Damage
            block$corrected[(pt+1):nrow(block)] <- 1
          }
        }
      } else {
        
        block$corrected <- block$Crop_Damage
      }
    } else {
      
      block$corrected <- block$Crop_Damage
      
    }
  }
  
  return(block)
}

# the automated version of the above function
end_of_block_correction_automated <- function(block) {
  
  if(nrow(block) == 1){
    
    block$corrected <- block$Crop_Damage
    
  }else{
    
    cd <- unique(block$CD[!is.na(block$CD)])
    if(length(cd) == 1 & is.na(block$CD[nrow(block)])){
      
      if(cd == 0){
        
        k <- block$Number_Doses_Week
        k_diff <- diff(k)
        is_reverse <- sum(k_diff < 0)
        if(is_reverse == 0){
          
          block$corrected <- block$Crop_Damage
        }else{
          
          L = nrow(block)
          reverse_indices <- which(k_diff < 0)
          if(L == 2 & min(reverse_indices) == 1){
            
            block$corrected <- block$Crop_Damage
            block$corrected[2] <- 1
            
          }else{
            
            HL <- L/2
            r1 <- reverse_indices[reverse_indices < HL]
            r2 <- reverse_indices[reverse_indices >= HL]
            if(length(r2) == 0){
              
              block$corrected <- block$Crop_Damage
            }else {
              
              cpt <- min(r2)
              block$corrected <- block$Crop_Damage
              block$corrected[(cpt+1):nrow(block)] <- 1
              
            }
            
          }
          
        }
      } else {
        
        block$corrected <- block$Crop_Damage
      }
    } else {
      
      block$corrected <- block$Crop_Damage
      
    }
  }
  
  return(block)
}

# checks if there is any non-integer values between 0 and 1 from the
# impute_intermittent function present in the block
btwn_zero_one <- function(block){
  
  entries <- sum(block$Crop_Damage > 0 & block$Crop_Damage < 1, na.rm = TRUE)
  return(entries)
  
}

# if there are values in the block where the previous present train entry is 0
# and the next train entry is 1, observe the Number_Doses_Week pattern and
# make adjustments to the values like 0.5, 0.75 to make them either 0 or 1
adjust_zero_ones <- function(block){
  
  dp <- which(block$Crop_Damage > 0 & block$Crop_Damage < 1)
  p1 <- min(dp) - 1
  p2 <- max(dp) + 1
  df <- block[p1:p2, ]
  nd <- df$Number_Doses_Week
  diff_nd <- diff(nd)
  
  if(sum(diff_nd < 0) == 0){
    
    return(block)
  } else{
    ind <- min(which(diff_nd < 0))
    df$Crop_Damage[1: ind] <- 0
    df$Crop_Damage[(ind + 1):nrow(df)] <- 1
    dfID <- df$ID
    block <- block[!block$ID %in% dfID,]
    block <- rbind(block, df)
    block <- block[order(block$ID),]
    return(block)
  }
}

# apply the imputation algorithm to any block of size > 2
treat_block <- function(bl){
  
  if(nrow(bl) < 3){
    
    cat("\n block length: ", nrow(bl), "..... SKIPPING!")
    if(nrow(bl) > 0) {
      
      df <- bl[, c("ID", "Estimated_Insects_Count", "Soil_Type", "Number_Doses_Week","Crop_Damage", "predicted")]
      df$block_length <- rep(nrow(bl), nrow(bl))
      
    }
    
  }else {
    
    bl <- takeFirst(bl)
    bl <- takeLast(bl)
    to_predict <- findTest(bl)
    
    if(length(to_predict) != 0){
      for(index in to_predict){
        
        bl <- impute_intermittent(bl, index)
        
      }
      
      check_confusion <- btwn_zero_one(bl)
      if(check_confusion > 0){
        
        bl <- adjust_zero_ones(bl)
        
      }
      
      df <- bl[, c("ID", "Estimated_Insects_Count", "Soil_Type", "Number_Doses_Week","Crop_Damage", "predicted")]
      df$block_length <- rep(nrow(bl), nrow(bl))
      
    }else{
      
      df <- bl[, c("ID", "Estimated_Insects_Count", "Soil_Type", "Number_Doses_Week","Crop_Damage", "predicted")]
      df$block_length <- rep(nrow(bl), nrow(bl))
      
    }
  }
  
  return(df)
}


#Final model run
#Final model run
#Final model run

require(readr)
require(caret)
require(xgboost)

setwd("F:/AV/Last man standing")
set.seed(2135)
source("./source.R")

load("./alldata_LMS.RData")

train <- alldata[alldata$trainFlag == 1,]
test <- alldata[alldata$trainFlag == 0,]

train$int_1 <- train$Estimated_Insects_Count/train$total_dosage
train[is.na(train)] <- -1
train[train == Inf] <- -1
train[train == -Inf] <- -1

test$int_1 <- test$Estimated_Insects_Count/test$total_dosage
test[is.na(test)] <- -1
test[test == Inf] <- -1
test[test == -Inf] <- -1

feature.names <- names(train) [!names(train) %in% c("ID", "Crop_Damage", "trainFlag", "Season_cropType", "Season_soilType", "cropType_soilType")]




############################# RUN THE XGB MODEL #############################
param <- list(objective = "multi:softprob",
              num_class = 3,
              max_depth = 8,
              eta = 0.01,
              subsample = 0.7,
              colsample_bytree = 0.8,
              min_child_weight = 40,
              max_delta_step = 3,
              gamma = 0.3,
              eval_metric = "mlogloss")

dtrain <- xgb.DMatrix(data = data.matrix(train[, feature.names]), label = train$Crop_Damage)
watchlist <- list(train = dtrain)

clf <- xgb.train(params = param,
                 data = dtrain,
                 nround = 900,
                 print.every.n = 20,
                 watchlist = watchlist)

tpreds <- predict(clf, data.matrix(test[, feature.names]))
tmat <- get_proba(tpreds, Id = test$ID)
write_csv(tmat, "./XGB_Commit1_test.csv")
s_test <- get_predict(tmat, isTest = TRUE)
write_csv(s_test, "./XGB_COMMIT_1.csv")                   # public LB: 84.9%

##################################################################################
########################## POST PROCESS MODEL OUTPUTS ############################
##################################################################################

train <- read_csv("train.csv")
train$predicted <- NA
test <- read_csv("test.csv")
pred <- read_csv("./XGB_COMMIT_1.csv")
names(pred)[2] <- "predicted"
test$Crop_Damage <- NA
test <- merge(test, pred)

md <- rbind(train, test)
md <- md[order(md$ID),]

############### intermittent values correction ###############
output <- data.frame(ID = character(), Estimated_Insects_Count = numeric(), Soil_Type = numeric(), Number_Doses_Week = numeric(),
                     Crop_Damage = numeric(), predicted = numeric(), block_length = numeric())

i = 1

while(i < nrow(md)) {
  
  rm(block, block1, block2, block3, block4)
  cat("\nBlock starting at row: ", i)
  ret <- findBlock(i)
  block <- ret$block
  
  if(nrow(block) < 3) {
    
    cat("\n Block length: ", nrow(block), "..... SKIPPING!")
    df <- block[, c("ID", "Estimated_Insects_Count", "Soil_Type", "Number_Doses_Week","Crop_Damage", "predicted")]
    df$block_length <- rep(nrow(block), nrow(block))
    output <- rbind(output, df)
    
  }else{
    # find the sub-blocks and mini-blocks inside a block
    # with varying Crop_Type and Soil_Type
    ct <- length(unique(block$Crop_Type))
    
  
    if(ct == 1){
      
      block1 <- block[block$Soil_Type == 0,]
      block2 <- block[block$Soil_Type == 1,]
      if(nrow(block1) > 0){
        
        output <- rbind(output, treat_block(block1))
      }
      
      if(nrow(block2) > 0){
        
        output <- rbind(output, treat_block(block2))
        
      }
      
      
    }
    else{
      
      block1 <- block[block$Crop_Type == 0 & block$Soil_Type == 0,]
      block2 <- block[block$Crop_Type == 0 & block$Soil_Type == 1,]
      block3 <- block[block$Crop_Type == 1 & block$Soil_Type == 0,]
      block4 <- block[block$Crop_Type == 1 & block$Soil_Type == 1,]
      
      if(nrow(block1) > 0){
        
        output <- rbind(output, treat_block(block1))
      }
      
      if(nrow(block2) > 0){
        
        output <- rbind(output, treat_block(block2))
      }
      
      if(nrow(block3) > 0){
        
        output <- rbind(output, treat_block(block3))
      }
      
      if(nrow(block4) > 0){
        
        output <- rbind(output, treat_block(block4))
      }
      
    }
  }
  
  i <- ret$ns
  
}

tid <- test$ID
test_corrected_v1 <- output[output$ID %in% tid,]
setdiff(tid, test_corrected_v1$ID)
a <- md[md$ID == "F00155944",][c("ID", "Estimated_Insects_Count", "Soil_Type", "Number_Doses_Week","Crop_Damage", "predicted")]
a$block_length <- 1
test_corrected_v1 <- rbind(test_corrected_v1, a)

write_csv(output, "./output94p.csv")
write_csv(test_corrected_v1, "./test_corrected_v1.csv")

############## the above code alone gave me 94.2% on the LB ##########

# Next part is to correct the end of block entries where 
# The blocks end with a test entry and the last train 
# entry in the block has Crop_Damage = 0
########## find the blocks ending with NAs ########
i <- 1
count <- 0
idx <- c()
while(i < nrow(md)) {
  
  ret <- findBlock(i)
  bl <- ret$block
  
  k <- 0
  if(length(unique(bl$Soil_Type)) == 2){
    
    b1 <- bl[bl$Soil_Type == 0,]
    k <- nrow(b1)
    
  }
  
  na_check = FALSE
  if(k != 0){
    
    na_check <- is.na(bl$Crop_Damage[nrow(bl)]) | is.na(bl$Crop_Damage[k])
  }else{
    
    na_check <- is.na(bl$Crop_Damage[nrow(bl)])
  }
  
  if(nrow(bl) >= 3 & na_check){
    
    count <- count + 1
    idx <- c(idx, i)
    cat("\n Match found at row:", i)
  } 
  
  i <- ret$ns
}

################## correct the takeLast function falacies of handing NAs in the end of time blocks ################
o2 <- data.frame(ID = character(), Estimated_Insects_Count = numeric(), Soil_Type = numeric(), Number_Doses_Week = numeric(),Crop_Damage = numeric(), 
                 predicted = numeric(), block_length = numeric(), corrected = numeric())


tmp94 <- read_csv("output94p.csv")
ss <- tmp94
dd <- rbind(train[c("ID", "Crop_Type", "Crop_Damage")], test[, c("ID", "Crop_Type", "Crop_Damage")])
names(dd)[3] <- "CD"
ss <- merge(ss, dd)

for(index in 1:length(idx)){
  
  cat("\nTreating block starting at row: ", idx[index])
  
  err_block <- findBlock2(idx[index])
  ct <- length(unique(err_block$Crop_Type))
  if(ct == 1){
    
    block1 <- err_block[err_block$Soil_Type == 0,]
    block2 <- err_block[err_block$Soil_Type == 1,]
    
    if(nrow(block1) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block1))
    }
    
    if(nrow(block2) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block2))
    }
    
    
  }else {
    
    block1 <- err_block[err_block$Crop_Type == 0 & err_block$Soil_Type == 0,]
    block2 <- err_block[err_block$Crop_Type == 0 & err_block$Soil_Type == 1,]
    block3 <- err_block[err_block$Crop_Type == 1 & err_block$Soil_Type == 0,]
    block4 <- err_block[err_block$Crop_Type == 1 & err_block$Soil_Type == 1,]
    
    if(nrow(block1) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block1))
    }
    
    if(nrow(block2) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block2))
    }
    
    if(nrow(block3) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block3))
    }
    if(nrow(block4) > 0){
      
      o2 <- rbind(o2, end_of_block_correction(block4))
    }
    
  }
  
}


tc94 <- read_csv("./test_corrected_v1.csv")
tc94 <- merge(tc94, o2[, c("ID", "corrected")], all.x = TRUE)
tc94$revised <- ifelse(is.na(tc94$corrected), tc94$Crop_Damage, tc94$corrected)

tc94$final <- ifelse(is.na(tc94$revised), tc94$predicted,
                     ifelse(tc94$revised > 0 & tc94$revised < 1, 0,
                            ifelse(tc94$revised > 1 & tc94$revised < 2, 1, tc94$revised)))


###### save the submissions #########

sub <- tc94[, c("ID", "final")]
names(sub)[2] <- "Crop_Damage"
write_csv(sub, "./XGB_Commit1_modified_version14.csv") # final submission - public LB: 96.4%, private LB: 96.09%




