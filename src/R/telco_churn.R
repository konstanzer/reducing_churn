library(readr)
customer_details <- read_csv("data/customer_details.csv")

#we only care about month-to-month customers churning
monthly_customers <- subset(customer_details, contract_type == 0)
attach(monthly_customers)
monthly_customers$churn <- ifelse(churn=='Yes',1,0)

library(fastDummies)
#create dummy variables for categorical vars with 3+ categories
monthly_customers <- dummy_cols(monthly_customers,
              select_columns = c('payment_type','phone_service','internet_service'))

#feature engineering tenure
monthly_customers$tenure <- as.integer(total_charges/monthly_charges)
monthly_customers$has_internet <- ifelse(internet_service==0,0,1)
attach(monthly_customers)
monthly_customers$adjusted_monthly = ifelse(has_internet,
                                            monthly_charges*.85,
                                            monthly_charges)
colnames(monthly_customers)
monthly_customers <- subset(monthly_customers, select=c(3,10:15, 18:19, 21:25))
length(complete.cases(monthly_customers))

churn_count = sum(monthly_customers$churn == 1)
churn_ratio = churn_count / length(churn)

#Logistic regression
library(Metrics)
set.seed(36)
colnames(monthly_customers)
data = subset(monthly_customers, select=c(2,11:14,4))
#stand.X = as.data.frame(scale(X))
test_size = 1:as.integer(length(churn) * .2)
train = data[-test_size,]
test = data[test_size,]

glm.fit = glm(churn ~ monthly_charges+tenure, data = train)
summary(glm.fit)

glm.probs=predict(glm.fit, train)
glm.pred=rep(0, length(train$churn))
glm.pred[glm.probs > .5] = 1
table(glm.pred, train$churn)
accuracy(glm.pred, train$churn)
#wow, I think this is actually recall
#predict more churn (less false neg) and it goes up
precision(glm.pred, train$churn)
recall(glm.pred, train$churn)

#test data
glm.probs=predict(glm.fit, test)
glm.pred=rep(0, length(test$churn))
glm.pred[glm.probs > .3] = 1
sum(glm.pred)/775
table(glm.pred, test$churn)
accuracy(glm.pred, test$churn)
precision(glm.pred, test$churn)
recall(glm.pred, test$churn)

glm.pred

#make a prediction
new <- data.frame(tenure = 55, internet_service_2 = 1)
ifelse(predict(glm.fit, new) > .3, 'Yes', 'No')

#discount to all fiber optic customers to 49
#discount to all DSL customers to 39
#revenue lost
sum(monthly_customers$internet_service_1
    & monthly_customers$churn == 0)
sum(monthly_customers$internet_service_2
    & monthly_customers$churn == 0)

loss = 829 * (52-39) + 966 * (70-49)
loss
new_monthly_cost = mean(monthly_customers$monthly_charges)*.75
new_customers_needed = loss/new_monthly_cost
#retain 38 percent of 
new_churn_ratio = 1 - (new_customers_needed + sum(monthly_customers$churn == 0)) /length(churn)
churn_ratio
#churn ratio must fall to 27 percent (16 diff) to breakeven
new_churn_ratio
#what do you predict the new churn ration will be?
#change monthly_charges in model reduced 21 if FO and 13 if DSL
#retrain model and note change in percentage predicted churn at .5 threshold

#total charges was the strongest predictor, followed by internet service vars
#and electronic check
#positively correlated with churn: senior, electronic check
#negatively correlated with churn: 0 or 1 phone lines,
#                           DSL or no internet, tenure
#An new old-person customer who uses e-checks
#      and multiple phone lines and fiber optic internet is a churner
#Recommendation: offer new customers a discount bundle for multiple phones
#      and fiber optics

write.csv(monthly_customers, 'monthly_customers.csv')

#Visuals
library(plotly)
fig <- plot_ly(data=monthly_customers, x=~tenure, y=~monthly_charges, color=~churn)
fig

#churn rate is 70% when bill is over $75 in first 15 mos.
#all others have a churn rate of 36%
bill = 75
months = 15
new_with_high_bill <- subset(monthly_customers, monthly_charges > bill & tenure < months)
new_churn_count = sum(new_with_high_bill$churn == 1)
new_churn_ratio = new_churn_count / length(new_with_high_bill$churn)
new_churn_ratio
the_rest <- subset(monthly_customers, monthly_charges <= bill | tenure >= months)
rest_churn_count = sum(the_rest$churn == 1)
rest_churn_ratio = rest_churn_count / length(the_rest$churn)
rest_churn_ratio

#cost driver is internet, not so much phone
#now, 52 for DSL and 70 for fiber optic
#2 phones lines cost 25
#I recommend capping bills at 75 for 15 months
# $50 fiber optic and 32 for dsl
cor(monthly_customers)
summary(monthly_customers)

monthly_customers2 <- subset(customer_details, contract_type == 0)
library(plyr)
plans = count(monthly_customers2$internet_service)
plans
fig <- plot_ly(data=monthly_customers2,
               x=c("No internet", "DSL", "Fiber optic"),
               y=c(524,1224,2128),
               name="Internet Plans",
               type="bar")
fig



 
 