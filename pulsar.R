require(ggplot2)
require(e1071)
require(Hmisc)
require(pROC)
require(caret)
require(rsample)      # data splitting 
require(randomForest) # basic implementation
require(neuralnet)
require(ROCR)
require(nnet)
require(corrplot)
require(xgboost)
require(hrbrthemes)
#Load data
pulsar <- read.csv("pulsar_stars.csv", header = T)
names(pulsar) = c("mean_int_prof","std_int_prof","excess_kurt_int_prof",
                  "skew_int_prof","mean_dm_snr","std_dm_snr",
                  "excess_kurt_dm_snr","skew_dm_snr","class")



#EDA and Cleaning
pulsar$class = as.factor(pulsar$class)
apply(pulsar, 2, function(x) any(is.na(x))) # check for NA missing data
#no missing values
head(pulsar, 3)
str(pulsar)

ggplot(pulsar) +
  aes(x = excess_kurt_int_prof, y = excess_kurt_dm_snr, colour = class) +
  geom_point(size = 1.06) +
  scale_color_viridis_d(option = "viridis") +
  labs(x = "Excess Kurtosis of Integrated Profile", y = "Excess Kurtosis of DM SNR Curve", title = "Pulsars") +
  theme_modern_rc() +
  facet_wrap(vars(class))

ggplot(pulsar) +
  aes(x = mean_int_prof, y = mean_dm_snr, colour = class) +
  geom_point(size = 1.06) +
  scale_color_viridis_d(option = "viridis") +
  labs(x = "Mean of Integrated Profile", y = "Mean of DM SNR Curve", title = "Pulsars") +
  theme_modern_rc() +
  facet_wrap(vars(class))

#Just based on these 4 variables the data looks seperable upon first glance...




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


pulsar$class = as.numeric(pulsar$class)-1
correlation_matrix = cor(pulsar)
corrplot(correlation_matrix, method="circle", bg = "grey", title = "Correlation Matrix")
cor.test(pulsar$std_int_prof, pulsar$excess_kurt_dm_snr)
pulsar$class = as.factor(pulsar$class)







## Split Data into Test and Train Sets

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

svm_predictions = predict(svmfit1, newdata = test)
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
plot(bestmod, data = pulsar, formula  = mean_int_prof~std_int_prof)
plot(ROC1, col = "blue")
title("Best SVM Model AUC")
AUC1 <- auc(ROC1)
AUC1
#better! but 0.9029 is not nearly as good as .966!



############### Random Forest Time  #############################
rf1 = randomForest(class~., data = train)
fr_pred_test = predict(rf1, type = "response", newdata = test)
auc(test$class, as.numeric(fr_pred_test), plot = T)




############# Quick Nerual Network  ####################
nn <- nnet(class~., data = train, size = 2)
nn_pred = predict(nn, newdata = test)
plot(nn)
ROC2 <- roc(test$class, as.numeric(nnet_pred))
plot(ROC2, col = "purple")
title("Single Layer NN Model AUC")
AUC2 <- auc(ROC2)
AUC2



############ Extreme Gradient Boosting  #####################
# class_numeric = as.numeric(train$class) - 1
# train_numeric = train
# train_numeric$class = class_numeric
# bstDense <- xgboost(data = as.matrix(train_numeric), label = train_numeric$class, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
# 
# 
# test_matrix = test
# test_matrix$class = as.numeric(test$class) - 1
# test_matrix = as.matrix(test_matrix)
# 
# xg_predictions =  predict(bstDense, newdata = as.matrix(test_matrix))
# auc(response = test$class, xg_predictions, plot = T)
# 
