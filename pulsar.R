library(ggplot2)
library(e1071)
library(Hmisc)
library(pROC)
library(caret)
library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(neuralnet)
library(ROCR)
library(nnet)
library(corrplot)
library(xgboost)

#Load data
setwd("D:/Kaggle/Pulsar")  #instert your own path here
pulsar <- read.csv("pulsar_stars.csv", header = T)
names(pulsar) = c("mean_int_prof","std_int_prof","excess_kurt_int_prof",
                  "skew_int_prof","mean_dm_snr","std_dm_snr",
                  "excess_kurt_dm_snr","skew_dm_snr","class")



#EDA
apply(pulsar, 2, function(x) any(is.na(x))) # check for NA missing data
head(pulsar, 3)
str(pulsar)
correlation_matrix = cor(pulsar)
corrplot(correlation_matrix, method="circle", bg = "grey")
cor.test(pulsar$std_int_prof, pulsar$excess_kurt_dm_snr)


ggplot(pulsar,aes(x = mean_int_prof, y = excess_kurt_int_prof)) + 
  geom_point(na.rm = T, aes(color = mean_dm_snr)) + geom_smooth()
#lets make a scatter plot to see the correlation visually
ggplot(data = pulsar, aes(x=std_dm_snr, y=excess_kurt_dm_snr))+
  geom_point()
#lets perform a f(x) = 1/x transformation to make the data linear...
ggplot(data = pulsar, aes(x=1/std_dm_snr, y=excess_kurt_dm_snr))+
  geom_point()+
  geom_smooth(method="lm")
#polynomial might give us a better fit
ggplot(data = pulsar, aes(x=1/std_dm_snr, y=excess_kurt_dm_snr))+
  geom_point()+
  geom_smooth(method="lm", formula = y ~ poly(x,3))




#split data into test and train sets
pulsar$class = as.factor(pulsar$class)
test_ind = sample(1:(nrow(pulsar)/10)) #10% of the data is in the test set
train = pulsar[-test_ind,]
test = pulsar[test_ind,]


#Lets try logistic regression first...
logistic_model =  glm(class~., data = train, family = "binomial")
summary(logistic_model)
#It seems a bit odd that excess kurtosis and skew of the DM SNR curve are not significant predictors of pulsars,
#especially considering that the mean and standard deviation are highly significant.
p1 = predict(logistic_model, type = "response", newdata = test)
auc(test$class, p1, plot = T)   #sensitivity = TP rate  specificity = TN rate
#.966 AUC is excellent!  We have a good balance of high specificity and sensitivity with our prediction on the test set
#Let's see what other models can achieve



###########################    Support Vector Machine    ####################
svmfit1 <- svm(class ~ ., data = train, kernel = "linear")
plot(svmfit1, train, mean_int_prof~std_int_prof, svSymbol = 8, dataSymbol = 1, symbolPalette = rainbow(2),
     color.palette = topo.colors)

svm_predictions = predict(svmfit3, newdata = test)
ROC1 <- roc(test$class, as.numeric(svm_predictions)-1)
plot(ROC1, col = "blue")
AUC1 <- auc(ROC1)
AUC1
# Area under the curve: 0.8867  Not nearly as good as logistic regression!  Let's try to tune our SVM

obj = tune.svm(class ~ ., data=pulsar, gamma = 1)
#the tune function does a 10 fold cross validation for us!  Wooo!
summary(obj)
bestmod = obj$best.model
bestmod
#here is our best model, radial kernel it is!
svm_predictions = predict(bestmod, newdata = test)
ROC1 <- roc(test$class, as.numeric(svm_predictions)-1)
plot(bestmod, data = pulsar, formula  =)
plot(ROC1, col = "blue")
title("Best SVM Model AUC")
AUC1 <- auc(ROC1)
AUC1
#better! but 0.9029 is not nearly as good as .966!



############### Random Forest Time  #############################
rf = randomForest(class~., data = train)
fr_pred_test = predict(rf1, type = "response", newdata = test)
auc(test$class, as.numeric(fr_pred_test), plot = T)
#Area under the curve: 0.9126



############# Quick Nerual Network  ####################
nnet1 <- nnet(class~., data = train, size = 2)
nnet_pred = predict(nnet1, type = "class", newdata = test)
plot(nnet1)
auc(test$class, as.numeric(nnet_pred_test), plot = T)
#Area under the curve: 0.9104




############ Extreme Gradient Boosting  #####################
class_numeric = as.numeric(train$class) - 1
train_numeric = train
train_numeric$class = class_numeric
bstDense <- xgboost(data = as.matrix(train_numeric), label = train_numeric$class, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")


test_matrix = test
test_matrix$class = as.numeric(test$class) - 1
test_matrix = as.matrix(test_matrix)

xg_predictions =  predict(bstDense, newdata = test_matrix, "response")
auc(response = test$class, xg_predictions, plot = T)

