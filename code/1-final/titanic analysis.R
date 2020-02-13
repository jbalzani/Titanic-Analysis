####
library(titanic)
library(caret)
library(tidyverse)
library(rpart)

#3 significant digits
options(digits = 3)

#clean the data - titanic_train is loaded with titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), #NA age to median age
         FamilySize = SibSp + Parch + 1) %>% #count family members 
  select(Survived, Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

#### above code from HarvardX staff
#### some question wording from HarvardX staff

#split titanic_clean into test and training sets
#set seed to 42, then use caret package to create a 20% data partition based on 
#survived column. 

#correct variable types
titanic_clean <- titanic_clean %>%
  mutate(Sex = as.factor(Sex),
         Pclass = as.factor(Pclass))

#create test and train data
y <- titanic_clean$Survived
set.seed(42, sample.kind = "Rounding")
test_index <- createDataPartition(y, times = 1, p = 0.20, list = FALSE)
titanic_test <- titanic_clean[test_index,]
titanic_train <- titanic_clean[-test_index,]

#what proportion of individuals in training set survived?
train_survived <- titanic_train %>% filter(Survived == 1)
nrow(train_survived)/nrow(titanic_train)
mean(titanic_train$Survived == 1) #better code

#for each individual in test set, randomly guess whether that person
#survived or not by sampling from the vector c(0, 1), assuming equal chance of surviving
#or not. what is the accuracy?
set.seed(3, sample.kind = "Rounding")
survival_guess <- sample(c(0, 1), nrow(titanic_test), replace = TRUE)
mean(survival_guess == titanic_test$Survived)
#note: if you put prob = c(x,1-x) argument in sample fcn, it changes the way the 
#random numbers are generated

#use the training set to determine whether members of a given sex were more likely
#to survive or die. apply this insight to generate survival predications on the test set
#what proportion of training set females survived?
train_fm <- titanic_train %>%
  filter(Sex == "female")
mean(train_fm$Survived == 1)

#what proportion of training set males survived?
train_m <- titanic_train %>%
  filter(Sex == "male")
mean(train_m$Survived == 1)

#alternate way of doing it
titanic_train %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1))

#predict survival by sex on the test set. if the survival rate for a given sex is over 0.5,
#predict survival for all members of that sex. if the survival rate is below 0.5, predict
#death 
#what is the accuracy of this sex-based prediction?
survival_guess_sex <- ifelse(titanic_test$Sex == "female", 1, 0)
mean(survival_guess_sex == titanic_test$Survived)

#predicting survival by class
#in which classes were passengers more likely to survive than die?
titanic_train %>%
  group_by(Pclass) %>%
  summarize(survival_rate = mean(Survived == 1))

#predict survival by class on test set. if survival rate for a class is over 0.5, predict
#survival, otherwise predict death
survival_pred_class <- ifelse(titanic_test$Pclass == 1, 1, 0)
mean(survival_pred_class == titanic_test$Survived)

#predict survival by both sex and class
#which combinations of sex and class were more likely to survive than die?
titanic_train %>%
  group_by(Sex, Pclass) %>%
  summarize(survival_rate = mean(Survived == 1))

#predict survival by both sex and class on the test set. predict survival if survival rate
#over 0.5, death otherwise
survival_pred_sex_class <- ifelse(titanic_test$Sex == "female" & 
                                    titanic_test$Pclass == 1, 1, 
                                  ifelse(titanic_test$Sex == "female" & 
                                           titanic_test$Pclass == 2, 1, 0))
#alternate way
#survival_pred_sex_class <- ifelse(titanic_test$Sex == "female" &
#                                    titanic_test$Pclass != 3, 1, 0)
#mean(survival_pred_sex_class == titanic_test$Survived)

#confusion matrix
#create confusion matrices for sex model, class model, and combined sex and class model
#need to convert predictions and survival status to factors
survival_pred_sex <- as.factor(survival_guess_sex)
survival_pred_class <- as.factor(survival_pred_class)
survival_pred_sex_class <- as.factor(survival_pred_sex_class)
confusionMatrix(data = survival_pred_sex, reference = titanic_test$Survived)
confusionMatrix(data = survival_pred_class, reference = titanic_test$Survived)
confusionMatrix(data = survival_pred_sex_class, reference = titanic_test$Survived)

#official code
# confusionMatrix(data = factor(survival_guess_sex),
#                 reference = titanic_test$Survived)
# confusionMatrix(data = factor(survival_pred_class),
#                 reference = titanic_test$Survived)
# confusionMatrix(data = factor(survival_pred_sex_class),
#                 reference = titanic_test$Survived)

#use F_meas function to calculate F1 scores for the sex model, class model, and combined
#model. which model has highest F1 score?
F_meas(data = factor(survival_guess_sex), reference = titanic_test$Survived)
F_meas(data = factor(survival_pred_class), reference = titanic_test$Survived)
F_meas(data = factor(survival_pred_sex_class), reference = titanic_test$Survived)

#set seed to 1. train a model using linear discriminant analysis with caret package
#using fare as the only predictor
#what is the accuracy on the test set?
set.seed(1, sample.kind = "Rounding")
lda_algo <- train(Survived~Fare, method = "lda", data = titanic_train)
survival_fare_lda <- predict(lda_algo, newdata = titanic_test)
mean(survival_lda == titanic_test$Survived)

#do the same using quadratic discriminant analysis
set.seed(1, sample.kind = "Rounding")
qda_algo <- train(Survived~Fare, method = "qda", data = titanic_train)
survival_preds_fare_qda <- predict(qda_algo, newdata = titanic_test)
mean(survival_preds_fare_qda == titanic_test$Survived)

#set seed to 1. train logistic regression with caret glm method using age as only 
#predictor.
#what is accuracy on test set?
set.seed(1, sample.kind = "Rounding")
reg_age_logistic <- train(Survived~Age, method = "glm", data = titanic_train)
survival_pred_age <- predict(reg_age_logistic, newdata = titanic_test)
mean(survival_pred_age == titanic_test$Survived)

#do the same with 4 predictors, age, sex, class, fare
set.seed(1, sample.kind = "Rounding")
reg_age_class_fare_sex <- train(Survived~Age + Pclass + Fare + Sex, method = "glm",
                                data = titanic_train)
pred_age_class_fare_sex <- predict(reg_age_class_fare_sex, newdata = titanic_test)
mean(pred_age_class_fare_sex == titanic_test$Survived)