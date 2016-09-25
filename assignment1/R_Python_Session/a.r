###############################################################################
##Part-1
setwd("/home/raju/Downloads/kaggle")
train <- read.csv("~/Downloads/kaggle/train.csv")
test <- read.csv("~/Downloads/kaggle/test.csv")
str(train)
head(train,100)

t1<-train[1:100,]
t2<-train[100:891,]
str(t1)
str(t2)
train <- t2
View(train)
table(train$Survived)
prop.table(table(train$Survived))
table(train$Age)
summary(train$Age)


##Checking the structure
str(test)
##Adding a Column
test$Survived <- rep(0,418)
##Checking the frequency of values in a column in a table
table(test$Survived)
str(test)
test$k <- rep(0)
str(test)

table(test$Survived)
##dropping a column
test$k<-NULL
str(test)
##extracting columns from table to be output use data.frame
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
##writing to csv file
write.csv(submit, file = "result.csv")


###############################################################################
##Part-2
prop.table(table(train$Sex, train$Survived))
#checking proportions in the table
prop.table(table(train$Sex, train$Survived),1)

##adding cnditions
test$Survived[test$Sex == 'female'] <- 1
table(test$Survived)

train$child<-rep(0)
train$child[train$Age<18]<-1
prop.table(table(train$child))

test$child<-rep(0)
test$child[test$Age<18]<-1
prop.table(table(test$child))

##aggregate
aggregate(Survived ~ child + Sex, data=train, FUN=sum)
aggregate(Survived ~ child + Sex, data=train, FUN=length)
aggregate(Survived ~ child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

train$familysize<-train$child+train$SibSp+1
str(train)


train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'
prop.table(table(train$Fare2))
prop.table(table(train$Fare2,train$Survived),1)

str(test)
test$Fare2 <- '30+'
test$Fare2[test$Fare < 30 & test$Fare >= 20] <- '20-30'
test$Fare2[test$Fare < 20 & test$Fare >= 10] <- '10-20'
test$Fare2[test$Fare < 10] <- '<10'


aggregate(Survived ~ Fare2 +  Sex + child, data=train, FUN=function(x) {sum(x)/length(x)})

###############################################################################
##Fitting Models 
## Initial
fit1<-lm(Survived ~ Sex,data=train)
summary(fit1)
coefficients(fit1) # model coefficients
#confint(fit, level=0.95) # CIs for model parameters
fitted(fit1) # predicted values
residuals(fit1) # residuals
sum(residuals(fit1))
#anova(fit1) # anova table
#vcov(fit1) # covariance matrix for model parameters

## With child and Fare and Sex
fit <- lm(Survived ~ Fare2 + Sex + child, data=train)
summary(fit)
coefficients(fit) # model coefficients
#confint(fit, level=0.95) # CIs for model parameters
fitted(fit) # predicted values
residuals(fit) # residuals
sum(residuals(fit))
anova(fit) # anova table
vcov(fit) # covariance matrix for model parameters
#influence(fit) # regression diagnostics 

test$Survived1<-round(predict(fit1,test))
test$Survived<-round(predict(fit,test))
submit1 <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived1)
##writing to csv file
write.csv(submit1, file = "result1.csv")
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
##writing to csv file
write.csv(submit, file = "result.csv")
##Plots



