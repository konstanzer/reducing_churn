telco <- read.csv('data/Kaggle_Telco.csv')
telco <- telco[complete.cases(telco),]
telco <- subset(telco, Contract == 'Month-to-month')
colnames(telco)
telco <- telco[,-c(16)]

telco$Churn <- ifelse(telco$Churn=='Yes',1,0)
churners = sum(telco$Churn)
churn_ratio = churners/length(telco$Churn)#43%

#Kaggle notebook: https://www.kaggle.com/farazrahman/telco-customer-churn-logisticregression
library(ggplot2)
ggplot(telco, aes(y=tenure, x=Churn)) + geom_boxplot()
ggplot(telco, aes(y=MonthlyCharges, x=Churn)) + geom_boxplot()
ggplot(telco, aes(y=TotalCharges, x=Churn)) + geom_boxplot()


#Clean categorical features
telco <- data.frame(lapply(telco, function(x) {
  gsub("No phone service", "No", x)}))
telco <- data.frame(lapply(telco, function(x) {
  gsub("No internet service", "No", x)}))

#Standardizing Continuous features
num_columns <- c("tenure", "MonthlyCharges", "TotalCharges")
telco[num_columns] <- sapply(telco[num_columns], as.numeric)
telco_int <- telco[,c("tenure", "MonthlyCharges", "TotalCharges")]
telco_int <- data.frame(scale(telco_int))

#Binning tenure
telco$tenure_bins <- as.factor(tapply(telco$tenure,
                                      cut(telco$tenure, 6)))
ggplot(telco, aes(tenure_bins)) + geom_bar() + xlab("Years")

#Dummies
library(fastDummies)
colnames(telco)
telco_cat <- telco[, -c(1,6,18:19)]
telco_cat <- dummy_cols(telco_cat, remove_selected_columns = TRUE,
                        remove_first_dummy = TRUE)

names(telco_cat)[names(telco_cat)=="InternetService_Fiber optic"] <- "InternetService_Fiber.optic"
names(telco_cat)[names(telco_cat)=="PaymentMethod_Electronic check"] <- "PaymentMethod_Electronic.check"
final <- cbind(telco_int, telco_cat)

#Splitting data
set.seed(0)
ix = 1:(3875/5)
val = final[ix,]
train = final[-ix,]

#Logistic regression
glm.fit = glm(Churn_Yes ~ ., data = train, family = "binomial")
summary(glm.fit)

library(MASS)
#Stepwise variable selection
glm.fit2 <- stepAIC(glm.fit, direction="both")
summary(glm.fit2)

#Variance Inflation Factor(VIF) is a measure of multicollinearity
#A predictor having a VIF of 2 or less is generally considered safe
library(car)
vif(glm.fit2)

glm.fit3 = glm(Churn_Yes ~ SeniorCitizen_1 + 
                 MultipleLines_Yes +
                 DeviceProtection_Yes +
                 TechSupport_Yes +
                 StreamingTV_Yes +
                 StreamingMovies_Yes +
                 PaperlessBilling_Yes +
                 PaymentMethod_Electronic.check, data = train, family = "binomial")
summary(glm.fit3)

#Evaluation
pred <- predict(glm.fit, type = "response", newdata = val)
summary(pred)
val$prob <- pred
pred_churn <- factor(ifelse(pred >= 0.50, "Yes", "No"))
actual_churn <- factor(ifelse(val$Churn_Yes==1,"Yes","No"))
table(actual_churn,pred_churn)

#Decision tree
library(rpart)
library(rpart.plot)

final$Churn_Yes <- as.factor(final$Churn_Yes)

#Training
Dtree = rpart(train$Churn_Yes ~., data = train, method = "class")
summary(Dtree)
rpart.plot(Dtree)
#Predicting 
DTPred <- predict(Dtree,type = "class", newdata = val)
table(val$Churn_Yes, DTPred)
(377+147)/775


#Fiber optic only
#Customers churn at a high rate the first month regardless of price
#We don't care about them, they'll eave anyway
fiber<-subset(final, InternetService_Fiber.optic == 1) #& tenure > -.85)
dsl<-subset(final, InternetService_Fiber.optic == 0 & InternetService_No == 0)

sum(dsl$Churn_Yes==1)
394/1223

length(fiber$Churn_Yes==1)
1162/2128
#685 new customers
2128*.322

#all internet customers
(394+1162)/(2128+1223)
(394+1162-685)/(2128+1223)
(churners-685)/3875

internet <- subset(telco,telco$InternetService == 'DSL')
mean(internet$MonthlyCharges)*685
2128*18

1694/5
ix = 1:340
val = fiber[ix,]
train = fiber[-ix,]

glm.fit = glm(Churn_Yes ~ MonthlyCharges + MonthlyCharges*tenure +
                tenure, data=train, family='binomial')
summary(glm.fit)
preds=predict(glm.fit, val) > .5
table(preds, val$Churn)
(158+50)/340


library(ggplot2)
ggplot(fiber, aes(x=MonthlyCharges, y=Churn_Yes)) + geom_point() + 
  stat_smooth(method="glm", method.args=list(family="binomial"), se=FALSE)

par(mar = c(4, 4, 1, 1)) # Reduce some of the margins so that the plot fits better
plot(fiber$tenure, fiber$Churn_Yes)
curve(predict(glm.fit, data.frame(tenure=x), type="response"), add=TRUE) 
