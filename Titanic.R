# This is an example of logistic regression using the Kaggle "Titanic" data.
# This regression does not contain any regularization. 
# For regularization, consider glmnet package.
library(Amelia) # for plotting a map of the NA values
library(ROCR) # for the ROC curve analysis
library(boot) # for the cross validation

# Exploratory analysis on the raw data
training.data.raw <- read.csv("train.csv", header=T, na.strings =c(""));
sapply(training.data.raw, function(x) sum(is.na(x)));
sapply(training.data.raw, function(x) length(unique(x)));
missmap(training.data.raw, main = "Missing values vs observed");

# Remove the variables that have too many NA values.
data<- subset(training.data.raw, select=c(2,3,5,6,7,8,10,12));
# Take care of the NA values.
# Replace the NA values in Age by the average of Age.
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm = T);

# contrasts shows how the variables have been dummyfied by R, and how to 
# interpret them in the model.
contrasts(data$Sex);
contrasts(data$Embarked);

# since there are only two missing values in Embarked, we will discard them
data <- data[!is.na(data$Embarked),];
rownames(data) <- NULL;

# Randomly pick 450 data as the training data. The rest is the test data.
set.seed(55)
trainNum <- sample(1:nrow(data), 450);
train <- data[trainNum,];
test <- data[setdiff(rownames(data), rownames(train)),];

# Perform logistic regression
model <- glm(Survived~., family=binomial(link="logit"),data=train);

#log(p/(1-p))=a*X1+b*X2+...+z*Xn, here for X1=Pclass, a=-1.20
summary(model);

# The wider the gap between Deviance (null model) and Resid. Dev (residual deviance),
# the better. 
# The variables which reduces the Resid. Dev significantly improve the model
# more. (The smaller the Resid. Dev, the better the model is.)
# A larger p-value indicates that the model without the variable explains
# the same amount of variation.
# Chi-square test is used to test the independece of two categorical variables
# from a single population.
anova(model, test = "Chisq");

# Assessing the predictive ability of the model
# The result of predict() is the probabilit of success, which is odds/(1+odds)
# since odds = p/(1-p).
fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response');
fitted.results <- ifelse(fitted.results > 0.5,1,0);
# MSE of the fit.
print(paste("The MSE of glm (randomly pick 450 data for training and the rest for test) is ",
                 mean((fitted.results-as.numeric(test$Survived))^2)));
misClasificError <- mean(fitted.results != test$Survived);
print(paste('The accuracy of the logistic regression is',1-misClasificError));

# Plot the ROC curve
p <- predict(model, newdata=subset(test, select=c(2,3,4,5,6,7,8)), type = "response");
# prediction is an object for ROC curve plotting.
pr <- prediction(p, test$Survived);
# "tpr" true positive rate (tp/(tp+fn), sensitivity, recall , power, 1-Type II error or probability of detection).
# It defines the ratio of correct positive results in all positive samples.
# "fpr" false positive rate (fp/(fp+tn), type I error, 1- Specificity).
# It defines the ratio of incorrect positive results in all negative samples.
# use ?ROCR::performance to check the list of available performance measures.
prf <- performance(pr, measure = "tpr", x.measure = "fpr");
plot(prf);

# Calculate the area underneath the ROC curve (AUC), which is the probability of the classifier
# will rank a randomly chosen positive sample higher than a randomly chose negative sample.
# A very nice illustration showing how to plot the ROC curve can be found here
# http://stats.stackexchange.com/questions/105501/understanding-roc-curve
# You can use AUC to select the threshold: the one with the smallest fpr and largest tpr.
auc<- performance(pr, measure = "auc");
print(paste("The area underneath the ROC curve (AUC) is ", auc@y.values[[1]]));
# auc@y.values[[1]];


# Run LOOCV cross validation 
glm.fit <- glm(Survived~., family=binomial(link="logit"),data=data);
# cv.glm$delta is the MSE of the model.
cv.err.LOOCV <- cv.glm(data, glm.fit);
print(paste("The prediction error using LOOCV is ", cv.err.LOOCV$delta[1]));
cv.err.K10 <- cv.glm(data, glm.fit, K=10);
print(paste("The prediction error using K-fold cv is ", cv.err.K10$delta[1]));