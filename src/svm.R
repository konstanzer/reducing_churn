#Support vector classifier
library(e1071)
library(Metrics)
set.seed(101)

customers <- read_csv('data/monthly_customers.csv')
colnames(customers)
x <- subset(customers, select=c(3,13))
y <- unlist(subset(customers, select=c(5)))
dat = data.frame(x=x, y=as.factor(y))

svm.fit = svm(y~., data=dat,
              kernel='linear', cost=.01, scale=FALSE)
plot(svm.fit, dat)
summary(svm.fit)

ypred = predict(svm.fit, dat)
table(predict=ypred, truth=y)
accuracy(ypred, y)

