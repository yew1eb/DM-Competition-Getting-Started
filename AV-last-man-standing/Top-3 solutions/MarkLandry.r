setwd("/Users/mark/Documents/AV_LastManStanding-2015/")
library(data.table)
train<-fread("Train_Fyxd0t8.csv")
test<-fread("Test_C1XBIYq.csv")
solFile<-fread("archive/gbm1_threshold50.csv")
cycleStart<-ifelse(c(3,train[,Pesticide_Use_Category][1:(nrow(train)-1)])-train[,Pesticide_Use_Category][1:nrow(train)]==-2,1,0)
midCycleStart<-ifelse(c(4000,train[,Estimated_Insects_Count][1:(nrow(train)-1)])-train[,Estimated_Insects_Count][1:nrow(train)]>2000,1,0)
cycles<-cumsum(cycleStart)
midCycles<-(cumsum(midCycleStart)-1)%%6
train[,whichMidCycle:=midCycles]
#par(mfrow=c(2,1))
#plot(cycles)
#plot(midCycles)
testcycleStart<-ifelse(c(3,test[,Pesticide_Use_Category][1:(nrow(test)-1)])-test[,Pesticide_Use_Category][1:nrow(test)]==-2,1,0)
testmidCycleStart<-ifelse(c(4000,test[,Estimated_Insects_Count][1:(nrow(test)-1)])-test[,Estimated_Insects_Count][1:nrow(test)]>2000,1,0)
testcycles<-cumsum(testcycleStart)
testmidCycles<-(cumsum(testmidCycleStart)-1)%%6
test[,whichMidCycle:=testmidCycles]
#par(mfrow=c(2,1))
#plot(testcycles)
#plot(testmidCycles)
#train[1:4000,plot(Pesticide_Use_Category)]
#test[1:4000,plot(Pesticide_Use_Category)]
#par(mfrow=c(2,1))
#train[1:4000,plot(Estimated_Insects_Count)]
#plot(insectDiffs[1:4000])
getDiffs<-function(x,shift=1){
  if(shift>0){return(x-x[c((1+shift):length(x),rep(length(x),shift))])}
  if(shift<0){return(x-x[c(rep(1,abs(shift)),1:(length(x)-abs(shift)))])}
}
train[,insectsPlus1:=getDiffs(Estimated_Insects_Count,1)]
#par(mfrow=c(1,1))
#plot(data.frame(train)[,8])
#colnames(train)[8]
train[,idNum:=as.numeric(substr(ID,2,nchar(ID)))]
test[,idNum:=as.numeric(substr(ID,2,nchar(ID)))]
train[,idNumPlus1:=getDiffs(idNum,1)]
train[,idNumPlus2:=getDiffs(idNum,2)]
train[,idNumPlus3:=getDiffs(idNum,3)]
train[,idNumPlus4:=getDiffs(idNum,4)]
train[,idNumPlus5:=getDiffs(idNum,5)]
train[,idNumPlus6:=getDiffs(idNum,6)]
train[,idNumMinus1:=getDiffs(idNum,-1)]
train[,idNumMinus2:=getDiffs(idNum,-2)]
train[,idNumMinus3:=getDiffs(idNum,-3)]
train[,idNumMinus4:=getDiffs(idNum,-4)]
train[,idNumMinus5:=getDiffs(idNum,-5)]
train[,idNumMinus6:=getDiffs(idNum,-6)]
test[,idNumPlus1:=getDiffs(idNum,1)]
test[,idNumPlus2:=getDiffs(idNum,2)]
test[,idNumPlus3:=getDiffs(idNum,3)]
test[,idNumPlus4:=getDiffs(idNum,4)]
test[,idNumPlus5:=getDiffs(idNum,5)]
test[,idNumPlus6:=getDiffs(idNum,6)]
test[,idNumMinus1:=getDiffs(idNum,-1)]
test[,idNumMinus2:=getDiffs(idNum,-2)]
test[,idNumMinus3:=getDiffs(idNum,-3)]
test[,idNumMinus4:=getDiffs(idNum,-4)]
test[,idNumMinus5:=getDiffs(idNum,-5)]
test[,idNumMinus6:=getDiffs(idNum,-6)]
train[,insectsPlus1:=getDiffs(Estimated_Insects_Count,1)]
train[,insectsPlus2:=getDiffs(Estimated_Insects_Count,2)]
train[,insectsPlus3:=getDiffs(Estimated_Insects_Count,3)]
train[,insectsPlus4:=getDiffs(Estimated_Insects_Count,4)]
train[,insectsPlus5:=getDiffs(Estimated_Insects_Count,5)]
train[,insectsPlus6:=getDiffs(Estimated_Insects_Count,6)]
train[,insectsMinus1:=getDiffs(Estimated_Insects_Count,-1)]
train[,insectsMinus2:=getDiffs(Estimated_Insects_Count,-2)]
train[,insectsMinus3:=getDiffs(Estimated_Insects_Count,-3)]
train[,insectsMinus4:=getDiffs(Estimated_Insects_Count,-4)]
train[,insectsMinus5:=getDiffs(Estimated_Insects_Count,-5)]
train[,insectsMinus6:=getDiffs(Estimated_Insects_Count,-6)]
train[,insectsAcc1:=getDiffs(insectsPlus1,1)]
train[,insectsAcc2:=getDiffs(insectsPlus1,2)]
train[,insectsAcc3:=getDiffs(insectsPlus1,3)]
train[,dosesPlus1:=getDiffs(Number_Doses_Week,1)]
train[,dosesPlus2:=getDiffs(Number_Doses_Week,2)]
train[,dosesPlus3:=getDiffs(Number_Doses_Week,3)]
train[,dosesPlus4:=getDiffs(Number_Doses_Week,4)]
train[,dosesPlus5:=getDiffs(Number_Doses_Week,5)]
train[,dosesPlus6:=getDiffs(Number_Doses_Week,6)]
train[,dosesMinus1:=getDiffs(Number_Doses_Week,-1)]
train[,dosesMinus2:=getDiffs(Number_Doses_Week,-2)]
train[,dosesMinus3:=getDiffs(Number_Doses_Week,-3)]
train[,dosesMinus4:=getDiffs(Number_Doses_Week,-4)]
train[,dosesMinus5:=getDiffs(Number_Doses_Week,-5)]
train[,dosesMinus6:=getDiffs(Number_Doses_Week,-6)]
train[,insects1and2:=insectsPlus1+insectsPlus2]
train[,insects2and3:=insectsPlus2+insectsPlus3]
train[,insects1and2and3:=insectsPlus1+insectsPlus2+insectsPlus3]
train[,doses1and2:=dosesMinus1+dosesMinus2]
train[,doses2and3:=dosesMinus2+insectsPlus3]
train[,doses1and2and3:=dosesMinus1+dosesMinus2+dosesMinus3]
test[,insectsPlus1:=getDiffs(Estimated_Insects_Count,1)]
test[,insectsPlus2:=getDiffs(Estimated_Insects_Count,2)]
test[,insectsPlus3:=getDiffs(Estimated_Insects_Count,3)]
test[,insectsPlus4:=getDiffs(Estimated_Insects_Count,4)]
test[,insectsPlus5:=getDiffs(Estimated_Insects_Count,5)]
test[,insectsPlus6:=getDiffs(Estimated_Insects_Count,6)]
test[,insectsMinus1:=getDiffs(Estimated_Insects_Count,-1)]
test[,insectsMinus2:=getDiffs(Estimated_Insects_Count,-2)]
test[,insectsMinus3:=getDiffs(Estimated_Insects_Count,-3)]
test[,insectsMinus4:=getDiffs(Estimated_Insects_Count,-4)]
test[,insectsMinus5:=getDiffs(Estimated_Insects_Count,-5)]
test[,insectsMinus6:=getDiffs(Estimated_Insects_Count,-6)]
test[,insectsAcc1:=getDiffs(insectsPlus1,1)]
test[,insectsAcc2:=getDiffs(insectsPlus1,2)]
test[,insectsAcc3:=getDiffs(insectsPlus1,3)]
test[,dosesPlus1:=getDiffs(Number_Doses_Week,1)]
test[,dosesPlus2:=getDiffs(Number_Doses_Week,2)]
test[,dosesPlus3:=getDiffs(Number_Doses_Week,3)]
test[,dosesPlus4:=getDiffs(Number_Doses_Week,4)]
test[,dosesPlus5:=getDiffs(Number_Doses_Week,5)]
test[,dosesPlus6:=getDiffs(Number_Doses_Week,6)]
test[,dosesMinus1:=getDiffs(Number_Doses_Week,-1)]
test[,dosesMinus2:=getDiffs(Number_Doses_Week,-2)]
test[,dosesMinus3:=getDiffs(Number_Doses_Week,-3)]
test[,dosesMinus4:=getDiffs(Number_Doses_Week,-4)]
test[,dosesMinus5:=getDiffs(Number_Doses_Week,-5)]
test[,dosesMinus6:=getDiffs(Number_Doses_Week,-6)]
test[,insects1and2:=insectsPlus1+insectsPlus2]
test[,insects2and3:=insectsPlus2+insectsPlus3]
test[,insects1and2and3:=insectsPlus1+insectsPlus2+insectsPlus3]
test[,doses1and2:=dosesMinus1+dosesMinus2]
test[,doses2and3:=dosesMinus2+insectsPlus3]
test[,doses1and2and3:=dosesMinus1+dosesMinus2+dosesMinus3]
########## This section is for tagging sequences and position therein
##########  everything is fast except the loop to put the sequence length
##########  in the records of all within that sequence, which takes 5 minutes
##########  because it's in a loop.
########## So this is the code run, and it has been saved out to a CSV to not redo it each time
#train[,isSame:=ifelse(abs(insectsMinus1) > 2,0,1)]
#train[,dupeOffset:=train[,isSame]*unlist(lapply(rle(train[,isSame])$lengths,seq_len))]
#train[,rowNum:=.I]
#train[,seqLength:=1]
#lenStarts<-train[dupeOffset==0,rowNum]
#start<-Sys.time()
#for(i in 1:length(lenStarts)){
#  if((lenStarts[i+1]-lenStarts[i])>1){
#  train$seqLength[lenStarts[i]:(lenStarts[i+1]-1)]<-(lenStarts[i+1]-lenStarts[i])}
#}
#end<-Sys.time(); end-start  ## Time difference of 4.911362 mins
#train[,toEndOffset:=seqLength-dupeOffset]
#
#test[,isSame:=ifelse(abs(insectsMinus1) > 2,0,1)]
#test[,dupeOffset:=test[,isSame]*unlist(lapply(rle(test[,isSame])$lengths,seq_len))]
#test[,rowNum:=.I]
#test[,seqLength:=1]
#lenStarts<-test[dupeOffset==0,rowNum]
#start<-Sys.time()
#for(i in 1:length(lenStarts)){
#  if((lenStarts[i+1]-lenStarts[i])>1){
#    test$seqLength[lenStarts[i]:(lenStarts[i+1]-1)]<-(lenStarts[i+1]-lenStarts[i])}
#}
#end<-Sys.time(); end-start  ## Time difference of 3.282757 mins (ran into a NULL value)
#test[,toEndOffset:=seqLength-dupeOffset]
#write.csv(train[,.(isSame,dupeOffset,rowNum,seqLength,toEndOffset)],"trainSequences.csv",row.names=F)
#write.csv(test[,.(isSame,dupeOffset,rowNum,seqLength,toEndOffset)],"testSequences.csv",row.names=F)
### Read static calculations (see above comment block)
trainSequences<-fread("trainSequences.csv")
testSequences<-fread("testSequences.csv")
train<-cbind(train,trainSequences)
test<-cbind(test,testSequences)
## see which works best: trying for 2's and still never using them; or just handling them like 1's
#train[,Crop_Damage:=pmin(Crop_Damage,1)]
#train<-train[Crop_Damage<2,]
library(h2o)
h2o.init(nthreads = -1,max_mem_size = '6G',assertion = F)
testHex<-as.h2o(test)
train[,Crop_Damage:=as.factor(Crop_Damage)]
trainHex<-as.h2o(train[1:80000,])
validHex<-as.h2o(train[80001:88858,])
gbmCounter<-gbmCounter+1
gbmName=paste0("gbm",gbmCounter)
modelCols<-colnames(test)[!(colnames(test) %in% c(
  "ID","midcycle","cycle","weeksDiffsFw","weeksDiffsBw","doses1and2and3","insects1and2and3"))]
g<-h2o.gbm(training_frame = trainHex,validation_frame = validHex,x=modelCols,y="Crop_Damage",model_id = gbmName,
           ntrees = 20000,max_depth = 10,stopping_rounds = 1,score_each_iteration = F,learn_rate = 0.02,
           stopping_tolerance = 0,col_sample_rate_per_tree = 0.8,col_sample_rate=0.7, sample_rate = 0.5)
pVal<-as.data.frame(h2o.predict(g,validHex))
p1<-as.data.frame(h2o.predict(g,testHex))
val<-data.table(cbind(train[80001:88858,],pVal))
val[,error:=ifelse(Crop_Type=="0",1-p0,p0)]
#par(mfrow=c(3,1))
#val[1300:1500,plot(ifelse(Crop_Damage=="0",error,-1*error))]
#val[1300:1500,plot(as.numeric(Crop_Damage))]
#val[1300:1500,plot(Estimated_Insects_Count)]
#par(mfrow=c(1,1)); val[1400:1450,plot(Estimated_Insects_Count)]; 
#val[1400:1450,.(insectsMinus1,Estimated_Insects_Count)]
#train[81400:81470,.(Estimated_Insects_Count,isSame,dupeOffset,seqLength,toEndOffset)]
#fullHex<-as.h2o(train,destination_frame = "full.hex")
#gF<-h2o.gbm(training_frame = fullHex,x=modelCols,y="Crop_Damage",model_id = paste0(gbmName,"_full"),
#           ntrees = 1800,max_depth = 4,learn_rate = 0.03,
#           col_sample_rate_per_tree = 0.7,sample_rate = 0.6)
#p2<-as.data.frame(h2o.predict(gF,testHex))
singleP1<-ifelse(p1$p2>0.65,"2",ifelse(p1$p0==pmax(p1$p0,p1$p1,p1$p2),"0",ifelse(p1$p1==pmax(p1$p0,p1$p1,p1$p2),"1","1")))
#singleP1<-ifelse(p2$p2>0.65,"2",ifelse(p2$p0==pmax(p2$p0,p2$p1,p2$p2),"0",ifelse(p2$p1==pmax(p2$p0,p2$p1,p2$p2),"1","1")))
#singleP1<-ifelse(p1$p1>0.5,"1","0")
solFile$Crop_Damage<-singleP1; 
table(solFile[,Crop_Damage])
solFile[1:10,]
write.csv(solFile,paste0(gbmName,".csv"),row.names=F)
dim(solFile)
#rf<-h2o.randomForest(training_frame = trainHex,x=modelCols,y="Crop_Damage",model_id = "rf1")
#summary(rf)
#dl<-h2o.deeplearning(training_frame = trainHex,nfolds=10,x=modelCols,y="Crop_Damage",model_id = "dl2",
#           activation = "RectifierWithDropout",hidden_dropout_ratios = c(0.2,0.2,0.2),stopping_rounds = 5,
#           stopping_tolerance = 0,epochs=50,hidden=c(512,512,512))
#p1<-as.data.frame(h2o.predict(h2o.getModel("dl2_cv_1"),testHex))
