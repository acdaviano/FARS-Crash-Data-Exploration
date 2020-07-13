#load the FARS excell file into a base file 
library(readxl)
accident <- read_excel("acc_aux_name1.xlsx")

#explore the data by looking at the summary, structure, head, tail, and view to ensure it 
#loaded properly, then make sure there are no missing variables to continue
View(accident)
summary(accident)
str(accident)
dim(accident)
head(accident)
tail(accident)
ma = sum(is.na(accident))
str(ma)

#once see that there are no missing values and the data has loaded properly with we proceed 
#we also note that we are ok with all of the names so we can press on without
#having to change them, next we install the tidyverse package to help with exploration
install.packages("tidyverse")
library(tidyverse)

#next we transform the dataset and turn the outcome variable which will be Ped_Fatal into
#a factor variable for the analysis 
accident$Ped_Fatal<-as.factor(ifelse(accident$Ped_Fatal == 1, 1, 0))

#Then we will remove Ped_inv to ensure there is no skewing of the analysis due to its 
#presence in the model. We will not include Ped_inv due to it being a collider/confounder 
#and needs to be present for the Ped_Fatal to happen.
accident<-subset(accident, select= -c(Ped_inv,Bycicle_inv,Bycicle_fatal))

#create a second file to work with 
accident0<-accident

#filter out the Texas only observations
accident1<-subset(accident0, STATE==48)

#look at the new structure and summary to make sure we got all the variables we wanted 
summary(accident1)
summary(accident1$Ped_Fatal)
str(accident1)
dim(accident1)
head(accident1,10)

#create a second file to work with 
accident2<-accident1

#create a training and test set of data in order to run the models for Ped_Fatal
#then create 2 list where the outcome values from the ML algorithms will live 
#for ACC and AUC
install.packages("caret")
library(caret)
set.seed(2010)
inTrainRows201 <- createDataPartition(accident2$Ped_Fatal,p=0.7,list=FALSE)
trainData201 <- accident2[inTrainRows201,]
testData201 <-  accident2[-inTrainRows201,]
nrow(trainData201)/(nrow(testData201)+nrow(trainData201))
AUC201 = list()
Accuracy201 = list()

#we have decided to do 5 ML alrorithms with the data to ensure we choose the right one for
#the data. We will look at a Logistic regression, a random forest, a boosted tree model
#with tuning and a grid included, a gbm (stochastic gradient boosting method) model, 
#and an SVM, all to see which one is better for the classification model to proceed 
#and then do a logistic regression for our overall outcome and to explain the pedestrian 
#deaths a bit better in vehicle crashes in TX

#after creating the list that all of the outputs for ACC and AUC will live in we build 
#the Logistic regression prediction model for Ped_Fatal
set.seed(2011)
logRegModel201 <- train(Ped_Fatal ~ ., data=trainData201, method = 'glm', family = 'binomial')
logRegPrediction201 <- predict(logRegModel201, testData201)
logRegPredictionprob201 <- predict(logRegModel201, testData201, type='prob')[2]
logRegConfMat201 <- confusionMatrix(logRegPrediction201, testData201[,"Ped_Fatal"])
str(logRegConfMat201)

#ROC Curve and Logistic regression for more important variables included to 
#predicting Ped_Fatal
library(pROC)
AUC201$logReg <- roc(as.factor(testData201$Ped_Fatal),as.numeric
                     (as.matrix((logRegPredictionprob201)))
                     )$auc
Accuracy201$logReg <- logRegConfMat201$overall['Accuracy']
Accuracy201$logReg
AUC201$logReg

#Random forest of values most important to Ped_Fatal
install.packages("randomForset")
library(randomForest)
set.seed(2012)
RFModel201 <- randomForest(Ped_Fatal ~ .,
                        data=trainData201,
                        importance=TRUE,
                        ntree=7000)

RFPrediction201 <- predict(RFModel201, testData201)
RFPredictionprob201 = predict(RFModel201,testData201,type="prob")[, 2]
RFConfMat201 <- confusionMatrix(RFPrediction201, testData201[,"Ped_Fatal"])

AUC201$RF <- roc(as.factor(testData201$Ped_Fatal),as.numeric
                 (as.matrix((RFPredictionprob201))))$auc
Accuracy201$RF <- RFConfMat201$overall['Accuracy'] 
AUC201$RF
Accuracy201$RF

#boosted tree model with a bit of tuning and an included grid search for Ped_Fatal
#install.packages("plyr") if not already through caret
#library(plyr)
set.seed(2013)
objControl201 <- trainControl(method='repeatedcv', number=10,  repeats = 10)
gbmGrid201 <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =20)
# run model
boostModel201 <- train(Ped_Fatal ~ .,data=trainData201, method='gbm',
                    trControl=objControl201, tuneGrid = gbmGrid201, verbose = FALSE)
# See model output to get an idea how it selects best model for Ped_Fatal
#trellis.par.set(caretTheme())
#plot(boostModel201)
boostPrediction201 <- predict(boostModel201, testData201)
boostPredictionprob201 <- predict(boostModel201, testData201, type='prob')[2]
boostConfMat201 <- confusionMatrix(boostPrediction201, testData201[,"Ped_Fatal"])

#ROC Curve
AUC201$boost <- roc(as.factor(testData201$Ped_Fatal),as.numeric
                    (as.matrix((boostPredictionprob201))))$auc
Accuracy201$boost <- boostConfMat201$overall['Accuracy']
AUC201$boost
Accuracy201$boost

#stochastic gradient boosting method that will help with automatic parameter selection of 
#Ped_Fatal
feature.names201=names(accident2)

for (f in feature.names201) {
  if (class(accident2[[f]])=="factor") {
    levels <- unique(c(accident2[[f]]))
    accident2[[f]] <- factor(accident2[[f]],
                       labels=make.names(levels))
  }
}
set.seed(301)
inTrainRows301 <- createDataPartition(accident2$Ped_Fatal,p=0.7,list=FALSE)
trainData301 <- accident2[inTrainRows301,]
testData301 <-  accident2[-inTrainRows301,]
fitControl201 <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary)
set.seed(3010)
gbmModel201 <- train(Ped_Fatal ~ ., data = trainData301,
                  method = "gbm",
                  trControl = fitControl201,
                  verbose = FALSE,
                  tuneGrid = gbmGrid201,
                  ## Specify which metric to optimize
                  metric = "ROC")
gbmPrediction201 <- predict(gbmModel201, testData301)
gbmPredictionprob201 <- predict(gbmModel201, testData301, type='prob')[2]
gbmConfMat201 <- confusionMatrix(gbmPrediction201, testData301[,"Ped_Fatal"])

#ROC Curve
AUC201$gbm <- roc(as.factor(testData301$Ped_Fatal),as.numeric
                  (as.matrix((gbmPredictionprob201))))$auc
Accuracy201$gbm <- gbmConfMat201$overall['Accuracy']
AUC201$gbm
Accuracy201$gbm

#SVM model next for Ped_Fatal outcome
install.packages("e1071")
library(e1071)
set.seed(2014)
svmModel201 <- train(Ped_Fatal ~ ., data = trainData301,
                  method = "svmRadial",
                  trControl = fitControl201,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction201 <- predict(svmModel201, testData301)
svmPredictionprob201 <- predict(svmModel201, testData301, type='prob')[2]
svmConfMat201 <- confusionMatrix(svmPrediction201, testData301[,"Ped_Fatal"])

#ROC Curve
AUC201$svm <- roc(as.factor(testData301$Ped_Fatal),as.numeric
                  (as.matrix((svmPredictionprob201))))$auc
Accuracy201$svm <- svmConfMat201$overall['Accuracy']
AUC201$svm
Accuracy201$svm

#Pool the results of all of the utilized models above to choose the best for selecting 
#variables for Ped_Fatal select the model with the higher AUC first then a high accuracy
#second.The higher AUC is more important due to having a binomial response variable and 
#the distribution of cases being skewed, so the AUC is better to choose the model due to 
#its averaging of the accuracy models it builds and places under the curve rather than 
# just the accuracy for this data.

set.seed(2016)
row.names201 <- names(Accuracy201)
col.names201 <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC201,Accuracy201),nrow = 5, ncol = 2,
                           dimnames = list(row.names201, col.names201))))

#In this case the gbm model was best followed by the boost model, those 2 models will
#move forward with overall analysis for the determination of the overall variables to 
#be used in the logistic regression to explain our outcome for the project
install.packages("gbm")
library(gbm)
install.packages("varImp")
library(varImp)

#looking at the important factors for Ped_Fatal with the gbm
gbmImp201 =caret::varImp(gbmModel201, scale = TRUE)
row201 = rownames(caret::varImp(gbmModel201, scale = TRUE)$importance)
#row201 = convert.names(row201)
rownames(gbmImp201$importance)=row201
plot(gbmImp201,main = 'Variable importance for Fatal Crash with Pedestrian prediction 
     with the Stochastic boosted tree')

#looking at the importance factors for Ped_Fatal with the boosted model
boostImp201 =caret::varImp(boostModel201, scale = TRUE)
row202 = rownames(caret::varImp(boostModel201, scale = TRUE)$importance)
#row202 = convert.names(row202)
rownames(boostImp201$importance)=row202
plot(boostImp201,main = 'Variable importance for Fatal Crash with Pedestrian prediction 
     with the boosted tree model')

#look at the base model of the Ped_Fatal, then look at the fitted model with the choices
#from the gbm importance plot since it was number one overall, I will include in the 
#fitted model some of the confirmations from the boosted model as well, however, they
#are both very similar and do not have very different importance for the same variables
mygbmlogit201base <- glm(Ped_Fatal ~  + 1,
                     data = accident1, family = "binomial" (link="logit"))

#look at the base model to establish if the fitted model we build is better
summary(mygbmlogit201base)

#looking at the coefficients for the entire log reg of the Fatal Crash with Pedestrian 
#logistic regression just to see what the full model's coefficients are for later
summary(logRegModel201)$coeff
summary(logRegModel201)

#then look at the logistic regression glm model with the variables that we want
#We will start from the top 10 most important variables and although 2 distinct 
#breaks in the graph points is sufficient for the choice of predictors, 
#we want to go down possibly 3 breaks to include the top 10 most important
#variables as chosen by the gbm and boost model. We will create a model where the gbm and 
#boost agree on the included variables since the same variables are in the top 10 with the 
#final say going to the gbm model on top 10 variable inclusion.
mygbmlogit201 <- glm(Ped_Fatal ~ road_depart + manner_of_collision  
                     + relation_to_road + Rollover + rural_or_urban + Pos_BAC + 
                   Speeding_inv + Hit_and_Run + TOD + Motocycle_inv,
                data = accident1, family = "binomial" (link="logit"))

#look at the fitted model
summary(mygbmlogit201)

#We will next look at the confidence intervals of the model
confint(mygbmlogit201)

#Next, we will perform a stepwise selection to ensure we are getting the best fitted model
#from the overall fitted model (original with 10 varaibles) above as a confirmation of 
#the included varaibles. We include the ful stepwise regression with every variable
#to try to reproduce the variable importance plot, then the stepwise with the chosen 10
#this will let us know if there are any more exclusions of the varaibles from the 10.
library(MASS)
logRegModel401 <- glm(Ped_Fatal ~ ., data=accident1, family = 'binomial' (link = "logit"))
thestepmodel201 <- logRegModel401 %>% stepAIC(trace = FALSE)
thestepmodel202 <- mygbmlogit201 %>% stepAIC(trace = FALSE)
summary(thestepmodel201)
summary(thestepmodel202)

#After seeing what variables are best for the overall model of prediction for the project
#we confirm that the fitted model we built with the identified most important variables 
#is the best fit model and no variables are further excluded from the elite set we chose.

#this will help with the amount of decimal places we see to get a better view of outputs
options(scipen=20)

#The following code looks at the odds ratios of the coefficients for the overall final model
Odds201<-exp(cbind(Odds_Ratio_Peddeathvsnofatal=coef(mygbmlogit201), confint(mygbmlogit201)))
Odds201
Odds401<-exp(cbind(Odds_Ratio_Peddeathvsnofatal=coef(thestepmodel201), 
                   confint(thestepmodel201)))
Odds401