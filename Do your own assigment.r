if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(riskCommunicator)) install.packages("riskCommunicator", repos = "http://cran.us.r-project.org")
if(!require(party)) install.packages("party", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")

library(riskCommunicator)
library(tidyverse)
library(caret)
library(party)
library(kernlab)

data("framingham")
set.seed(10)
# we aim to identify factors which allow prediction of myocardial infarctions

# data preparation:
select_columns = c('SEX', 
                   'TOTCHOL',
                   'AGE',
                   'SYSBP',
                   'DIABP',
                   'BMI',
                   'BPMEDS',
                   'HEARTRTE',
                   'GLUCOSE',
                   'MI_FCHD',
                   'PREVCHD',
                   'PREVSTRK',
                   'PREVMI'
                   )

dataset_cleaned = framingham[select_columns]

dataset_cleaned = dataset_cleaned %>% mutate(MI_FCHD_fact = as.factor(MI_FCHD))
# remove NA data
dataset_cleaned = na.omit(dataset_cleaned)

# start with general descriptors:
summary(dataset_cleaned)

# proportion with MI
mean(as.logical(dataset_cleaned$MI_FCHD))

# only 15 pct have MI, that means 85% accuracy is achievable by saying everyone is healthy.
# to even the proportions:
MI = dataset_cleaned[dataset_cleaned$MI_FCHD_fact == 1,]
normal = dataset_cleaned[dataset_cleaned$MI_FCHD_fact == 0,]
# sample normal so that it matches MI length
normal_sample = normal[sample(nrow(normal), nrow(MI)), ]
# join the df back together
dataset_cleaned = rbind(MI,normal_sample)

print(mean(as.logical(dataset_cleaned$MI_FCHD)))

# split train and test
test_index = createDataPartition(y = dataset_cleaned$MI_FCHD_fact, times = 1, p = 0.5, list = FALSE)
train_set = dataset_cleaned[-test_index,]
val_set = dataset_cleaned[test_index,]

# test factor dependence using linear modelling:

linear_model = lm(MI_FCHD ~. -MI_FCHD_fact, data = train_set)
summary(linear_model)

lm_prediction = predict(linear_model, val_set)
lm_prediction = as.numeric(lm_prediction > 0.5)
lm_accuracy = mean(lm_prediction == val_set$MI_FCHD_fact)

print(paste0('LM Accuracy:',lm_accuracy))

# train a decision tree:
tc = trainControl(method = "cv", number=10)
rt_fit = train(MI_FCHD_fact ~. -MI_FCHD, data = train_set, method = "ctree", trControl=tc, tuneLength=10)
plot(rt_fit)
print(rt_fit)

# test the accuracy
ctree_prediction = predict(rt_fit, val_set)
ctree_accuracy = mean(ctree_prediction == val_set$MI_FCHD_fact)
print(paste0('Descision Tree Accuracy:',ctree_accuracy))

# train a k nearest neighbour
tc = trainControl(method = "cv", number=10)
knn_fit = train(MI_FCHD_fact ~. -MI_FCHD, data = train_set, method = "knn", trControl=tc, tuneLength=10)

knn_prediction = predict(knn_fit, val_set)
knn_fit
plot(knn_fit)
knn_accuracy = mean(knn_prediction == val_set$MI_FCHD_fact)
print(paste0('KNN Accuracy:',knn_accuracy))

# train a random forest
tc = trainControl(method = "cv", number=5)
rf_fit = train(MI_FCHD_fact ~. -MI_FCHD, data = train_set, method = "rf", trControl=tc, tuneLength=10)
rf_fit
plot(rf_fit)
rf_prediction = predict(rf_fit, val_set)
rf_accuracy = mean(rf_prediction == val_set$MI_FCHD_fact)

# train a generalised linear model
tc = trainControl(method = "cv", number=10)
glm_fit = train(MI_FCHD_fact ~. -MI_FCHD, data = train_set, method = "glm", trControl=tc, tuneLength=10)
summary(glm_fit)
glm_prediction = predict(glm_fit, val_set)
glm_accuracy = mean(glm_prediction == val_set$MI_FCHD_fact)
print(paste0('Generalized Linear Model Accuracy:',rf_accuracy))

# use the model predictions to create an ensemble
# will have a voting system where majority rules
ct_vote = as.numeric(ctree_prediction)-1
knn_vote = as.numeric(knn_prediction)-1
rf_vote = as.numeric(rf_prediction)-1
glm_vote = as.numeric(glm_prediction)-1
lm_vote = as.numeric(lm_prediction)-1 

vote_tally = ct_vote+knn_vote+rf_vote+glm_vote+lm_vote
final_prediction = as.factor(as.numeric(vote_tally >= 3))
accuracy = mean(final_prediction == val_set$MI_FCHD_fact)
print(accuracy)

# final accuracy
print(paste0('LM Accuracy:',lm_accuracy))
print(paste0('Decision Tree Accuracy:',ctree_accuracy))
print(paste0('KNN Accuracy:',knn_accuracy))
print(paste0('Random Forest Accuracy:',rf_accuracy))
print(paste0('Generalized Linear Model Accuracy:',glm_accuracy))
print(paste0('Ensemble Accuracy:',accuracy))

guessing = function (g){
  random_guess = rbinom(nrow(val_set), 1, 0.5)
  random_guess_accuracy = mean(final_prediction == as.factor(random_guess))
}

monte_carlo = sapply(1:10000, guessing)
ggplot() + geom_histogram(aes(x = monte_carlo ), bins=50) + 
  geom_vline(aes(xintercept = accuracy, ), color='red') +
  annotate("text", x=accuracy+0.002, y=500, label="Ensemble Accuracy", angle=90) +
  annotate("text", x=0.5, y=500, label="guessing", color = 'white')