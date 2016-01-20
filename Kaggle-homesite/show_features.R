require(data.table)

train = fread('../input/train.csv',header=TRUE,data.table=F)

for(name in names(train)[1:100])
{
  print(name)
  if((typeof(train[,name])!="character"))
  {
    p0 <- hist(train[,name],plot = F)
    p1 <- hist(train[train[,"QuoteConversion_Flag"]==0,name],plot = F, breaks = p0$breaks)
    p2 <- hist(train[train[,"QuoteConversion_Flag"]==1,name],plot = F, breaks = p0$breaks)

    par(mfrow=c(1,2))

    plot( p1, col=rgb(0,0,1,1/4), main = name,xlab = "QuoteConversion_Flag=0")  # first histogram
    plot( p2, col=rgb(1,0,0,1/4), main = name, xlab = "QuoteConversion_Flag=1")  # second

    if(length(table(train[,name]))<100)
    {
      par(mfrow=c(1,1))
      avg = tapply(train[,"QuoteConversion_Flag"], train[,name], mean)
      sdev = tapply(train[,"QuoteConversion_Flag"], train[,name], sd) / sqrt(tapply(train[,"QuoteConversion_Flag"], train[,name], length))
      plot(1:length(avg), avg, main=name,ylab = "Average label value", type = "p",col="red")

      arrows(1:length(avg), avg-sdev, 1:length(avg), avg+sdev, length=0.05, angle=90, code=3)
    }
  }
}
