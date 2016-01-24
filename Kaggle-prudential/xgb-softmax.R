library(readr)
library(xgboost)
library(Metrics)

# I have not yet swept the parameters very well - probably a lot of room for improvement.
# Enjoy - jps

####################################################################################################
# FUNCTION / VARIABLE DECLARTIONS
####################################################################################################

# evaluation function that we'll use for "feval" in xgb.train...
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- ScoreQuadraticWeightedKappa(as.numeric(labels),as.numeric(round(preds)))
  return(list(metric = "kappa", value = err))
}

# declare these as variables: easier to reuse the script; and many are included in the output filename
myObjective       <- "multi:softmax"  # xgb parm... objective... multiclass classification
myBooster         <- "gbtree"         # xgb parm... type of booster... gbtree
myValSetPCT       <- 5.0              # pct of training set to hold for validation
myEta             <- 0.02             # xgb parm... smaller = more conservative
myGamma           <- 0.3              # xgb parm... bigger = more conservative
myMaxDepth        <- 15               # xgb parm... bigger = might overfit
mySubsample       <- 0.9              # xgb parm... 0.9 to 0.7 usually good
myColSampleByTree <- 0.7              # xgb parm... 0.5 to 0.7 usually good
myMinChildWeight  <- 3                # xgb parm... bigger = more conservative
myNRounds         <- 100              # xgb parm... bigger = might overfit
myEarlyStopRound  <- 10               # xgb parm... stop learning early if no increase after this many rounds
myNThread         <- 3                # num threads to use

####################################################################################################
# MAINLINE
####################################################################################################
set.seed(1234)

cat("read train and test data...\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

feature.names <- names(train)[2:ncol(train)-1]

# remove NA values...
train[is.na(train)] <- 0
test[is.na(test)]   <- 0

cat("replace text variables with numerics factors...\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

# response values are in the range [1:8] ... make it [0:7] for xgb softmax....
train$Response = train$Response - 1

cat("create dval/dtrain/watchlist...\n")
featureSet <- train[,feature.names]
vRowIds    <- sample(nrow(train),round(nrow(train)*myValSetPCT/100))
dval       <- xgb.DMatrix(data=data.matrix(featureSet[vRowIds,]),label=train$Response[vRowIds])
dtrain     <- xgb.DMatrix(data=data.matrix(featureSet[-vRowIds,]),label=train$Response[-vRowIds])
watchlist  <- list(val=dval,train=dtrain)

cat("running xgboost...\n")
param <- list(   objective           = myObjective,
                 booster             = myBooster,
                 eta                 = myEta,
                 max_depth           = myMaxDepth,
                 subsample           = mySubsample,
                 colsample_bytree    = myColSampleByTree,
                 min_child_weight    = myMinChildWeight,
                 gamma               = myGamma,
                 # num_parallel_tree   = 2
                 # alpha               = 0.0001,
                 # lambda              = 1,
                 num_class           = 8,
                 nthread             = myNThread
)

clf <- xgb.train(params              = param,
                 data                = dtrain,
                 nrounds             = myNRounds,
                 early.stop.round    = myEarlyStopRound,
                 watchlist           = watchlist,
                 feval               = evalerror,
                 maximize            = TRUE,
                 verbose             = 0
)

# just for keeping track of how things went...
# run prediction on training set so we can add the value to our output filename
trainPreds <- as.integer(round(predict(clf, data.matrix(train[,feature.names]))))
trainScore <- ScoreQuadraticWeightedKappa(round(trainPreds),train$Response)

outFileName <- paste("z0.00000 - ",trainScore,
                     " - ",clf$bestScore,
                     " - xgb - kappa - softmax",
                     " - ",myBooster,
                     " - ",myValSetPCT,
                     " - ",myEta,
                     " - ",myGamma,
                     " - ",myMaxDepth,
                     " - ",mySubsample,
                     " - ",myColSampleByTree,
                     " - ",myMinChildWeight,
                     " - ",myNRounds,
                     " - ",myEarlyStopRound,
                     " - ",clf$bestInd,".csv",sep = "")

cat("\ngenerate submission...\n")
submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,feature.names]))))

# we predicted in the range of [0:7] based on softmax... move back to [1:8]...
submission$Response <- submission$Response + 1

write_csv(submission, outFileName)