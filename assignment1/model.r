# Setting path of code directory
path = getwd()
setwd(path)

# configure multicore
#library(doMC)
#registerDoMC(cores=2)

# Reading given train data
trainData = read.csv("train_data.csv")
str(trainData)

# Data cleansing: Removing url column from data
modeldata = trainData[2:61]
str(modeldata)

# Spliting given data into train and test data for validation
library(caTools)
set.seed(2000)
splitData = sample.split(mo$shares, SplitRatio = 0.7)
train = subset(trainData, splitData==TRUE)
test = subset(trainData, splitData==FALSE)

# Linear regression model
#model = glm(shares ~ ., data=modeldata)
#summary(model)

# Linear regression model with less features
model = glm(shares ~ abs_title_sentiment_polarity + global_subjectivity + LDA_01 + 
              self_reference_min_shares + kw_avg_avg + kw_max_avg + kw_min_avg + kw_min_max + 
              data_channel_is_entertainment + data_channel_is_lifestyle + num_imgs + num_self_hrefs +
              num_hrefs + n_tokens_content + n_tokens_title + timedelta, data=modeldata)
#summary(model)

# Simple support vector regression model
#library(e1071)
#model = svm(shares ~ ., data=modeldata)
#summary(model)

# Random forest tree
#library(randomForest)
#set.seed(1)
#trainSmall = modeldata[sample(nrow(modeldata), 2000), ]
#model = randomForest(shares ~ ., data=modeldata)

# Support vector regression model with tuning and using best model
#library(e1071)
#tuneResult = tune(svm, shares ~ ., data = modeldata, ranges = list(epsilon = seq(0,1,0.2), cost = 2^(2:3))) 
#plot(tuneResult)
#model = tuneResult$best.model


##################################################################################
# Reading test data
testData = read.csv("test_data.csv")
str(testData)

# Data cleansing: removing url column
modelTest = testData[2:60]

# Prediction using model
predictTest = predict(model, newdata = modelTest, type = "response")

# Getting result as dataframe
result_df = data.frame(id = numeric(), shares = numeric())
for(i in 1:length(predictTest)){
  result_df = rbind(result_df, data.frame(id = i-1, shares = predictTest[i]))
}
str(result_df)

# Write result daraframe into csv file
write.csv(result_df, "output.csv", quote = FALSE, row.names = FALSE)