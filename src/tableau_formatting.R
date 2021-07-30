library(readr)
customer_details <- read_csv("data/customer_details.csv")
#we only care about month-to-month customers churning
monthly_customers <- subset(customer_details, contract_type == 0)
#drop id and contract type
monthly_customers <- subset(monthly_customers, select = -c(1,8))
attach(monthly_customers)
#features engineering
monthly_customers$tenure <- as.integer(total_charges/monthly_charges)
#formatting names for tableau
monthly_customers$payment_type <- ifelse(
  payment_type=='Credit card (automatic)','Credit card',
  monthly_customers$payment_type)
monthly_customers$payment_type <- ifelse(
  payment_type=='Bank transfer (automatic)','Bank transfer',
  monthly_customers$payment_type)

monthly_customers$phone_service <- ifelse(
  phone_service==0,'None',
  monthly_customers$phone_service)
monthly_customers$phone_service <- ifelse(
  phone_service==1,'One line',
  monthly_customers$phone_service)
monthly_customers$phone_service <- ifelse(
  phone_service==2,'Two-plus lines',
  monthly_customers$phone_service)

monthly_customers$internet_service <- ifelse(
  internet_service==0,'None',
  monthly_customers$internet_service)
monthly_customers$internet_service <- ifelse(
  internet_service==1,'DSL',
  monthly_customers$internet_service)
monthly_customers$internet_service <- ifelse(
  internet_service==2,'Fiber optic',
  monthly_customers$internet_service)

monthly_customers$is_senior_citizen <- ifelse(
  is_senior_citizen==0,'No','Yes')

monthly_customers$monthly_charges <- round(monthly_charges,
                                           digits=0)

write.csv(monthly_customers, 'tableau_churn.csv')

summary(monthly_charges)
med=median(monthly_charges)
mean=mean(monthly_charges)
churners = sum(churn=="Yes")
churn_ratio_above_median = sum(monthly_charges>med & churn=='Yes')/1938
churn_ratio_below_median = sum(monthly_charges<med & churn=='Yes')/1938
#53% churn if they pay above median, 33% churn if it's below
monthly_customers$has_internet <- ifelse(internet_service==0,0,1)
attach(monthly_customers)
monthly_customers$adjusted_monthly = ifelse(has_internet,
                                            monthly_charges*.85,
                                            monthly_charges)





