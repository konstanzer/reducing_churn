library(readr)
customer_details <- read_csv("data/customer_details.csv")
#we only care about month-to-month customers churning
monthly_customers <- subset(customer_details, contract_type == 0)
#drop id and contract type
monthly_customers <- subset(monthly_customers, select = -c(1,8))
attach(monthly_customers)
length(complete.cases(monthly_customers))

churn_count = sum(churn == 'Yes')
churn_ratio = churn_count / length(churn)

monthly_customers$churn <- ifelse(churn=='Yes',1,0)
monthly_customers$gender <- ifelse(gender=='Male',1,0)
monthly_customers$partner <- ifelse(partner=='Yes',1,0)
monthly_customers$dependents <- ifelse(dependents=='Yes',1,0)

library(fastDummies)
#create dummy variables for categorical vars with 3+ categories
monthly_customers <- dummy_cols(monthly_customers, select_columns = 'payment_type')
monthly_customers <- dummy_cols(monthly_customers, select_columns = 'phone_service')
monthly_customers <- dummy_cols(monthly_customers, select_columns = 'internet_service')

#drop extra dummy vars
monthly_customers <- subset(monthly_customers, select = -c(5:7, 14, 17, 20))
#feature engineering
monthly_customers$tenure <- total_charges/monthly_charges
#analysis showed gender, dependents, bank or cc transfer,
#and partnered aren't significant
#tenure made charges irrelevant
monthly_customers <- subset(monthly_customers, select = -c(1,3:6,8,9))


#Logistic regression
glm.fit = glm(churn~., data=monthly_customers)
summary(glm.fit)

glm.probs=predict(glm.fit)
glm.pred=rep(0, length(churn))
glm.pred[glm.probs>.5]=1
table(glm.pred, churn)

library(Metrics)
accuracy(glm.pred, monthly_customers$churn)

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



 
 