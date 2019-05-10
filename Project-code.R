############################### Poverty Prediction ######################################

if (!require("randomForest")) install.packages("randomForest")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("xgboost")) install.packages("xgboost")
if (!require("corrplot")) install.packages("corrplot")
if (!require("glmnet")) install.packages("glmnet")
if (!require("rpart")) install.packages("rpart")
if (!require("factoextra")) install.packages("factoextra")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("maboost")) install.packages("maboost")
if (!require("pROC")) install.packages("pROC")
if (!require("ROCR")) install.packages("ROCR")
if (!require("SDMTools")) install.packages("SDMTools")
if (!require("Hmisc")) install.packages("Hmisc")
if (!require("nnet")) install.packages("nnet")
if (!require("caret")) install.packages("caret")
if (!require("ISLR")) install.packages("ISLR")

library(randomForest)
library(dplyr)
library(tidyverse)
library(xgboost)
library(glmnet)
library(ISLR)
library(caret)
library(corrplot)
library(nnet)
library(Hmisc)
library(SDMTools)
library(ROCR)
library(pROC)
library(maboost)
library(rpart)
library(gridExtra)
library(factoextra)

#setwd("C:\\Users\\vidhy\\Downloads\\costa-rican-household-poverty-prediction")
povertyData<-read.csv("train.csv")

################################# Data Preprocessing ###################################

colnames(povertyData)[colSums(is.na(povertyData)) > 0]
sapply(povertyData, function(x) sum(is.na(x)))[colSums(is.na(povertyData)) > 0]
nullValueCols<-sapply(povertyData, function(x) sum(is.na(x)))

#### 1. Treating Null values
#### House -Rent
povertyDataV2Na <- povertyData[rowSums(is.na(povertyData['v2a1'])) > 0,]
houseOwnership=c(table(povertyDataV2Na$tipovivi1)[2],table(povertyDataV2Na$tipovivi2)[2],table(povertyDataV2Na$tipovivi3)[2],table(povertyDataV2Na$tipovivi4)[2])
barplot(houseOwnership,names.arg=c("Owned and paid","Owned-paying","Rented","Precarious"),
        col=c("light green","red","green","yellow"),las=0.5,main="Home ownership status for missing rents",xlab="House Ownership",ylab = "Frequency")

#### Number of tablets
povertyDataV18Na <- povertyData[rowSums(is.na(povertyData['v18q1'])) > 0,]
tabletOwnership=c(table(povertyDataV18Na$v18q))

#### Years behind in school

povertyDataVrez <- povertyData[rowSums(is.na(povertyData['rez_esc'])) == 0,]
summary(povertyDataVrez$age)

#### Convet all NAs to zeors
povertyData[is.na(povertyData)] <- 0

#### Check if all nulls are trated
colnames(povertyData)[colSums(is.na(povertyData)) > 0]

############################Feature Engineering###################################
#Removing all columns which are squared
povertyData <- povertyData[,-(134:140),drop=FALSE]
#povertyData <- povertyData[ , -which(names(povertyData) %in% c("female"))]

has_many_values <- function(x) n_distinct(x) > 1
dup_var <- function(x) lapply(x, c) %>% duplicated %>% which

#####################################################################################


#Aggregate individual variable to household level
povertyData <- povertyData %>%
  group_by(idhogar) %>%
  mutate(mean_age = mean(age, na.rm = TRUE)) %>%
  mutate(no_of_disabled = sum(dis)) %>%
  mutate(no_of_children = sum(estadocivil1)) %>%
  mutate(no_of_coupledunion = sum(estadocivil2)) %>%
  mutate(no_of_married = sum(estadocivil3)) %>%
  mutate(no_of_divorced = sum(estadocivil4)) %>%
  mutate(no_of_separated = sum(estadocivil5)) %>%
  mutate(no_of_widower = sum(estadocivil6)) %>%
  mutate(no_of_single = sum(estadocivil7)) %>%
  mutate(no_of_instlevel1 = sum(instlevel1)) %>%
  mutate(no_of_instlevel2 = sum(instlevel2)) %>%
  mutate(no_of_instlevel3 = sum(instlevel3)) %>%
  mutate(no_of_instlevel4 = sum(instlevel4)) %>%
  mutate(no_of_instlevel5 = sum(instlevel5)) %>%
  mutate(no_of_instlevel6 = sum(instlevel6)) %>%
  mutate(no_of_instlevel7 = sum(instlevel7)) %>%
  mutate(no_of_instlevel8 = sum(instlevel8)) %>%
  mutate(no_of_instlevel9 = sum(instlevel9)) %>%
  ungroup()

povertyData <- povertyData %>%
  #select(-tamviv) %>% # number of persons living in the household
  #select(-hogar_total) %>% # # of total individuals in the household
  #select(-r4t3) %>% # Total persons in the household
  #select(-tamhog) %>% # size of the household
  #select(-r4t1) %>% # persons younger than 12 years of age
  #select(-r4t2) %>% # persons 12 years of age and older
  #select(-agesq) %>% # Age squared
  select(-Id) %>% # removing id
  select(-idhogar)

#### Check if the dataset has any non numeric column(s)
povertyData %>%
  select_if(funs(!is.numeric(.)))
#### Recode values in dependency, edjefe, edjefa
povertyData[,c("dependency","edjefe","edjefa")] <- povertyData %>%
  select(dependency,edjefe,edjefa) %>%
  mutate_all(funs(ifelse(. == "yes",1,ifelse(. == "no",0,.)))) %>%
  mutate_all(as.numeric)
#### Train and test split
row<-nrow(povertyData)
set.seed(12345)
trainindex <- sample(row, row*.7, replace=FALSE)
training <- povertyData[trainindex, ]
validation <- povertyData[-trainindex, ]

train_labels <- as.numeric(training$Target) - 1
test_labels<-as.numeric(validation$Target)-1
train_data <- as.matrix(training[,-134])
test_data <- as.matrix(validation[,-134])

################################## Macro F1 score #####################################

f1_score <- function(predicted, expected, positive.class="1") {
  predicted <- factor(as.character(predicted), levels=unique(as.character(expected)))
  expected  <- as.factor(expected)
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  #### Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  #### Binary F1 or Multi-class macro-averaged F1
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
}

############################### Ridge and Lasso ######################################

#### Ridge Regression
ridge.mod = glmnet(train_data,train_labels,alpha=0, family="multinomial", type.multinomial="grouped")
#### Plotting the coefficents of all features against different lambda values
plot(ridge.mod,xvar="lambda")

#### cv.glmnet() - performs 10-fold cross-validation
ridge_cv=cv.glmnet(train_data,train_labels,alpha=0, family="multinomial", type.multinomial="grouped")
#### Plotting log(lambda) against multinomial deviance
plot(ridge_cv)

#### Select lamda that minimizes training MSE
bestlam=ridge_cv$lambda.min
bestlam #0.04901


#### making predictions using lambda.min
ridge.predicted<- predict(ridge.mod, s=bestlam, newx= test_data, type='class')
ridgeF1<-f1_score(ridge.predicted,test_labels) #0.146


#### Calculating the confusion matrix for ridge regression
confMat<-confusionMatrix(factor(ridge.predicted), factor(test_labels))
confMat #accuracy -0.6743

mean((as.numeric(ridge.predicted)-as.numeric(test_labels))^2) #1.03


#### The Lasso ( alpha=1)
lasso.mod = glmnet(train_data,train_labels,alpha=1, family="multinomial", type.multinomial="grouped")

#### Plotting the coefficents of all features against different lambda values
plot(lasso.mod, xvar="lambda")
summary(lasso.mod)

#### using cross validation
lasso_cv=cv.glmnet(train_data,train_labels,alpha=1, family="multinomial", type.multinomial="grouped")

#### Plotting log(lambda) against multinomial deviance
plot(lasso_cv)

bestlam_lasso=cv.out$lambda.min #0.004490
lasso.pred=as.numeric(predict(lasso.mod,s=bestlam, newx= test_data, type='class'))

#### Calculating the confusion matrix for Lasso regression
confMat<-confusionMatrix(factor(lasso.pred), factor(test_labels))
confMat #acc: 65.34%

lassoF1<-f1_score(lasso.pred,test_labels) #0.1505

#### Feature Selection using Lasso
x=model.matrix(Target~.,povertyData)
y=povertyData$Target

out=glmnet(x,y,alpha=1, family= "multinomial",lambda=grid)
#### lasso.coef has features for each target class
lasso.coef=predict(out,type="coefficients",s=bestlam)



################################# MA BOOST ########################################




#### The maboost and rpart library is used to run the model.
#### Initially the model is fit and predicited against the train dataset.

test1 <- as.data.frame(test_data)
fit_maboost <- maboost(Target ~ ., data=training, iter = 100 ,verbose = TRUE)

#### Predicting on the train set to check for overfitting
predict_maboost <- predict(fit_maboost, training, type = "response")
#### Predicting on the test set
predict_maboost_test <- predict(fit_maboost, test1, type = "response")  

head(predict_maboost)
score_maboost <- sum(training$Target == predict_maboost)/nrow(training)
score_maboost #0.6797


#### Calculating Macro F1 score
f1_score(predict_maboost_test,test_labels) #0.106



############################## Principal Component Analysis #############################



#### Initially a subset of the dataset is created by removing
#### the variable with missing value causing noise in the dataset.
Data <- subset( training, select = -elimbasu5 )
training1 <- training[-training$elimbasu5]
str(Data)

#### The below code helps us to fetch the correlation plot.
povertyDataCoVar <- cov(povertyData)
corrplot(povertyDataCoVar, method="circle",is.corr = FALSE)

#### eigen() to calculate the eigen values and eigen vectors
target.eigen <- eigen(povertyDataCoVar)
str(target.eigen)

which(apply(training, 2, var)==0)

#### We are using built in functions prcomp which uses spectral decomposition approach.

pcadata1 <- prcomp(Data, center = TRUE, scale = TRUE)
summary(pcadata1)

#### The below plot helps us to visualize the PCA plot between the four target variables
#### showing that there exists a significant difference between them.

fviz_pca_ind(pcadata1, geom.ind = "point", pointshape = 21,
             pointsize = 2,
             fill.ind = as.factor(Data$Target),
             col.ind = "black",
             palette = "jco",
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Poverty Levels") +
  ggtitle("2D PCA-plot of the dataset") +
  theme(plot.title = element_text(hjust = 0.5))




############################# Support vector Machines ##################################



#### traincontrol() - to control the computational nuances present in the train dataset
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#### The below SVM method uses Linear kernel. Preprocess function uses preprocess
#### parameter - center and scale and the tunelength which holds an integer value.

svm_Linear <- train(as.factor(Target) ~., data = Data, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)


test_pred <- predict(svm_Linear, newdata = test_data)

#### Macro f1 score - 0.128
f1_score(test_pred,test_labels)



############################ Multinomial Logistic Regression #############################




trainingSubset <- subset( training, select = -elimbasu5 )
validationSubset <- subset( validation, select = -elimbasu5 )

#### Building the multinomial logistic model with Target as y and remaining variables as x
multinomModel <- multinom(Target ~ ., data=trainingSubset)

#### Running the probabilistic model to check the weights/probabilities at each level.
#### This is not used for prediction
Multinomprob <- predict(multinomModel, data = validationSubset, type ="probs")
Multinompred <- predict(multinomModel, validationSubset)
validation_labels<-validationSubset$Target

#### Confusion-matrix
cm <- table(Multinompred, validation_labels)

#### Macro f1 score - 0.165
f1_score(Multinompred, validation_labels)

#### Wald -test for significance of variables based on the beta values
z <- summary(multinomModel)$coefficients/summary(multinomModel)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2

#### Second iteration -> removing the variables that Wald-test identified as insignificant
trainingSubset <- subset( trainingSubset, select = -edjefa )
validationSubset <- subset( validationSubset, select = -edjefa )
trainingSubset <- subset( trainingSubset, select = -agesq )
validationSubset <- subset( validationSubset, select = -agesq )

#Building the model with excluding the variables
multinomModel2 <- multinom(Target ~ ., data=trainingSubset)

#Prediction using the second model
Multinompred2_2 <- predict(multinomModel2, validationSubset)
#confusion matrix
cm <- table(Multinompred2_2, validation_labels)

#### Macro f1 score - 0.1699
f1_score(Multinompred2_2, validation_labels)




################################ Random Forest ######################################
#### Random forest model




rfModel<-randomForest(as.factor(Target)~. - elimbasu5, data=training,importance=TRUE, na.action=na.omit)

#### Variable Importance Plot to see the important predictors
varImpPlot(rfModel)

summary(rfModel)
predicted=predict(rfModel,validation,type="response")

#### Confusion Matrix
table(predicted,validation$Target)

error <- mean(validation$Target != predicted)
paste('Accuracy',round(1-error,4))

#### Macro f1 score - 0.4394
f1_score(predicted,validation$Target)

#### tuning parameters using Caret
RF = train(as.factor(Target) ~. - elimbasu5, data = training, method = "ranger", trControl = trainControl(method = "cv"),preProcess = c("center", "scale"))

predicted1=predict(RF,validation)

#### Confusion Matrix
table(predicted1,validation$Target)

error <- mean(validation$Target != predicted1)
paste('Accuracy',round(1-error,4))

#### Macro f1 score - 0.449
f1_score(predicted1,validation$Target)
#####################################################################################


