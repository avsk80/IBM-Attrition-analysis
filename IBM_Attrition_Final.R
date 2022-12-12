library(car)

data <- read.csv('trainset.csv')
head(data)

numeric = c('Age','DistanceFromHome','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','TotalWorkingYears',
            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Leaves',
            'Absenteeism')

## PCA
pca_data <- data[, numeric]
head(pca_data)

require(FactoMineR)

pca1 <- PCA(pca_data, ncp = 12)

pca1$eig

Correlation_Matrix=as.data.frame(round(cor(pca_data,pca1$ind$coord)**2*100,0))
Correlation_Matrix[with(Correlation_Matrix, order(-Correlation_Matrix[,1])),]

##-------Model Building----------------

##-------Forward Selection - All Features-----------------

# Complicated Model - All continuous features
attach(data)
data$Attrition <- ifelse(data$Attrition=='Yes', 1, 0)

# Intercept Model
mod1 <- glm(data$Attrition ~ 1, family = 'binomial')

# Complicated Model
mod2 <- glm(data$Attrition ~ 1 + Age + DistanceFromHome +MonthlyIncome + NumCompaniesWorked + PercentSalaryHike + TotalWorkingYears +
              TrainingTimesLastYear + YearsAtCompany + YearsSinceLastPromotion + YearsWithCurrManager + Leaves +
              Absenteeism, family = 'binomial')

# Forward Selection - BIC
step_1 <- step(mod1, data = Rateprof, direction = 'forward', scope = list(lower = mod1, upper = mod2), k=log(366))
step_1
summary(step_1)

##---------Forward Selection - PCA Features--------------

#col = ['TotalWorkingYears', 'NumCompaniesWorked', 'DistanceFromHome', 'Absenteeism', 'TrainingTimesLastYear']
# Intercept Model
mod3 <- glm(data$Attrition ~ 1, family = 'binomial')

# Complicated Model
mod4 <- glm(data$Attrition ~ 1 + DistanceFromHome + NumCompaniesWorked + TotalWorkingYears +
              TrainingTimesLastYear + Absenteeism, family = 'binomial')

# Forward Selection - BIC
step_2 <- step(mod1, data = Rateprof, direction = 'forward', scope = list(lower = mod3, upper = mod4), k=log(366))
step_2
summary(step_2)

vif(step_2)

## --------------Forward Selection - Base: Continuous, Complex: Continuous + Categorical-------------
mod5 <- glm(data$Attrition ~ 1 + DistanceFromHome + NumCompaniesWorked + TotalWorkingYears +
                     TrainingTimesLastYear, family = 'binomial')

mod6 <- glm(data$Attrition ~ 1+DistanceFromHome + NumCompaniesWorked + TotalWorkingYears +
              TrainingTimesLastYear+BusinessTravel+JobRole+MaritalStatus+OverTime+factor(JobInvolvement) + 
              factor(JobSatisfaction) + factor(StockOptionLevel), family = 'binomial')

# Forward Selection - BIC 
step_3 <- step(mod5, data = data, direction = 'forward', scope = list(lower = mod5, upper = mod6), k=log(366))
step_3
summary(step_3)

final_mod <- glm(data$Attrition ~ 1 + DistanceFromHome + NumCompaniesWorked + 
                   TotalWorkingYears + TrainingTimesLastYear + OverTime + factor(StockOptionLevel) + 
                   BusinessTravel + factor(JobInvolvement) + factor(JobSatisfaction), family = "binomial")
summary(final_mod)

vif(final_mod)

varImp(final_mod)

library(pscl)
pR2(final_mod)["McFadden"]

prob <- predict(final_mod, data[, c('DistanceFromHome', 'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 'OverTime',
                            'StockOptionLevel', 'BusinessTravel', 'JobInvolvement', 'JobSatisfaction')], type='response')
optimal <- optimalCutoff(data$Attrition, prob, optimiseFor = "Both")
optimal

data$prob <- prob
data$attr_pred <- ifelse(data$prob>=optimal, 1, 0)
confusionMatrix(data$Attrition, data$prob, threshold = optimal)

sensitivity(data$Attrition, data$prob, threshold = optimal)
specificity(data$Attrition, data$prob, threshold = optimal)
misClassError(data$Attrition, data$prob, threshold = optimal)

## Test set
test <- read.csv('testset.csv')
head(test)
test$Attrition <- ifelse(test$Attrition=='Yes', 1, 0)
test$prob <- predict(final_mod, test[, c('DistanceFromHome', 'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear', 'OverTime',
                                    'StockOptionLevel', 'BusinessTravel', 'JobInvolvement', 'JobSatisfaction')], type='response')
test$attr_pred <- ifelse(test$prob>=optimal, 1, 0)
confusionMatrix(test$Attrition, test$prob, threshold = optimal)

sensitivity(test$Attrition, test$prob, threshold = optimal)
specificity(test$Attrition, test$prob, threshold = optimal)
misClassError(test$Attrition, test$prob, threshold = optimal)