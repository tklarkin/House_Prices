

# Libraries and Directory -------------------------------------------------

#Libraries
library(caret)
library(doParallel)
library(plyr)
library(dplyr)

setwd("H:/tklarkin_backup/Documents/Kaggle/Housing")

# Loading Data ------------------------------------------------------------

training <- read.csv("./training.csv")
testing <- read.csv("./testing.csv")

# #Without DL features
# training <- training[,1:289]
# testing <- testing[,1:288]

response <- training$SalePrice
predictors <- subset(training, select = -c(Id, SalePrice))

# predictors <- predictors[1:500,1:30]
# response <- response[1:500]

# Metadata Generation Function --------------------------------------------

#Nested cross validation for metadata generation for base-learners
metadata.gen <- function(x, y, method, metric, trControl, tuneLength, verbose, ...) {
  preds <- NULL
  set.seed(10)
  cv = createFolds(y, k = trControl$number)
  # iterate over folds
  for(iter in 1:length(cv)) {
    # test indices
    if(verbose!= 0) {
      cat("Processing outer loop iteration", iter, "/", length(cv), "\n")
    }
    test_indices = cv[[iter]]
    train_indices = setdiff(seq(length(y)), test_indices)
    dat <- cbind.data.frame(y = y[train_indices], x[train_indices,])
    if(method %in% c("bartMachine", "bagEarthGCV")){
      model = train(y ~ ., data = dat, method=method,
                    metric=metric, trControl=trainControl(method = "none"), tuneLength=tuneLength,...)
    } else{
      model = train(y ~ ., data = dat, method=method,
                    metric=metric, trControl=trControl, tuneLength=tuneLength, ...)
    }
    # do predictions
    predictions = predict(model, x[test_indices,])
    # format predictions into form for summary Function
    tmp = data.frame(predictions, y[test_indices], 
                     test_indices, stringsAsFactors=FALSE)
    colnames(tmp) <- c("pred", "obs", "rowIndex")
    preds[[iter]] <- tmp
  } # end for(iter in 1:length(cv))
  preds = plyr::ldply(preds, .parallel = TRUE)
  #training model on all data
  dat <- cbind.data.frame(y = y, x)
  if(method %in% c("bartMachine", "bagEarthGCV")){
    model = train(y ~ ., data = dat, method=method,
                  metric=metric, trControl=trainControl(method = "none"), tuneLength=tuneLength,...)
  } else{
    model = train(y ~ ., data = dat, method=method,
                  metric=metric, trControl=trControl, tuneLength=tuneLength, ...)
  }
  return(list(metadata = preds, fit = model))
}

# Creating custom metric --------------------------------------------------

lrmseSummary <- function(data, lev = NULL, model = NULL) {
  require(MLmetrics)
  out <- RMSE(y_pred = log(data$pred), y_true = log(data$obs))
  names(out) <- "LRMSE"
  out
}

tl <- 10
metric <- "LRMSE"

# Settting Up Training ----------------------------------------------------

#Running in parallel
cl <- makeCluster(4)
registerDoParallel(cl)

set.seed(10)

seeds <- vector(mode = "list", length = 31) #length is = (n_repeats*nresampling)+1
for(i in 1:30) seeds[[i]]<- sample.int(n=1000, 36) #large enough for all par combos
seeds[[31]]<-sample.int(1000, 1)#for the last model

fitControl <- trainControl(method = "cv", number = 10, seeds = seeds,
                           allowParallel = TRUE, summaryFunction = lrmseSummary,
                           savePredictions = "final", search = "grid")


# Gathering the Metadata via 10-fold CV -----------------------------------

base.1 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.05)

base.2 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.10)

base.3 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.15)

base.4 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.20)

base.5 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.25)

base.6 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.30)

base.7 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.35)

base.8 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.40)

base.9 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                       metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                       tau = 0.45)

base.10 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.50)

base.11 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.55)

base.12 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.60)

base.13 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.65)

base.14 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.70)

base.15 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.75)

base.16 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.80)

base.17 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.85)

base.18 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.90)

base.19 <- metadata.gen(x = predictors, y = response, method = "rqlasso", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1,
                        tau = 0.95)

base.20 <- metadata.gen(x = predictors, y = response, method = "qrf", tuneLength = tl,
                        metric = metric, maximize = FALSE, trControl = fitControl, verbose = 1)


# Gathering Metadata ------------------------------------------------------

meta.list <- list(base.1$metadata, base.2$metadata, base.3$metadata, base.4$metadata, 
                  base.5$metadata, base.6$metadata, base.7$metadata, base.8$metadata, 
                  base.9$metadata, base.10$metadata, base.11$metadata, base.12$metadata, 
                  base.13$metadata, base.14$metadata, base.15$metadata, base.16$metadata,
                  base.17$metadata, base.18$metadata, base.19$metadata, base.20$metadata)

#Initializing data frame for metadata
metadata <- data.frame(matrix(ncol = length(meta.list), nrow = length(response)))

for(i in 1:length(meta.list)){
  metadata[,i] <- arrange(meta.list[[i]], rowIndex)[,"pred"]
}

colnames(metadata) <- paste0("mod.", 1:ncol(all.preds))

dim(metadata)
head(metadata[duplicated(lapply(metadata, summary))])
sum(is.na(metadata))
nearZeroVar(metadata)


# Creating Interactions ---------------------------------------------------


library(EBglmnet)

eben <- cv.EBglmnet(x = as.matrix(metadata),
                    y = response, prior = "elastic net", Epis = TRUE)

eblasso <- cv.EBglmnet(x = as.matrix(metadata),
                       y = response, prior = "lassoNEG", Epis = TRUE)

#List of interactions to include
interactions <- rbind(eben$fit[,1:2], eblasso$fit[,1:2])
interactions <- interactions[which(interactions[,1]-interactions[,2] != 0),]

#
train.features2 <- matrix(rep(0), ncol = nrow(interactions), nrow = nrow(metadata))
for(i in 1:nrow(interactions)){
  train.features2[,i] <- metadata[,interactions[i,1]]*metadata[,interactions[i,2]]
}

colnames(train.features2) <- paste0("meta.Interaction.", 1:ncol(train.features2))

int.metadata.train <- cbind.data.frame(metadata, train.features2)

# Creating t-SNE ----------------------------------------------------------

library(Rtsne)
library(earth)

dim <- 4

set.seed(123)
train.tsne.out <- as.data.frame(Rtsne(X = int.metadata.train,
                                      dim = dim, theta = 0, perplexity = 50)$Y)
colnames(train.tsne.out) <- paste0("meta.tsne.", 1:dim)

#Making parametric mapping with MARS
par.mapping <- earth(y = train.tsne.out, x = int.metadata.train ,
                     degree = 3, pmethod = "cv", nfold = 10)

# #Most important vars
# set.seed(123)
# imp.vars <- varImp(base.19$fit)$importance
# imp.vars <- imp.vars[order(imp.vars$Overall, decreasing = TRUE), , drop = FALSE]


# Implement Meta-learner --------------------------------------------------


training.metadata <- cbind.data.frame(int.metadata.train, train.tsne.out)


metalearner <- train(x = training.metadata, y = response, method = "cubist", metric = "LRMSE",
                      trControl = fitControl, maximize = FALSE,
                      tuneGrid = expand.grid(committees = c(1, 10, 20, 50, 100),
                                             neighbors = c(0, 1, 5, 9)), 
                      control = cubistControl(rules = 50))

metalearner2 <- train(x = training.metadata, y = response, method = "cubist", metric = "LRMSE",
                     trControl = fitControl, maximize = FALSE,
                     tuneGrid = expand.grid(committees = c(1, 10, 20, 50, 100),
                                            neighbors = c(0, 1, 5, 9)),
                     control = cubistControl(rules = 100))

# Making Predictions ------------------------------------------------------

pred.1 <- predict(base.1$fit, newdata = testing)
pred.2 <- predict(base.2$fit, newdata = testing)
pred.3 <- predict(base.3$fit, newdata = testing)
pred.4 <- predict(base.4$fit, newdata = testing)
pred.5 <- predict(base.5$fit, newdata = testing)
pred.6 <- predict(base.6$fit, newdata = testing)
pred.7 <- predict(base.7$fit, newdata = testing)
pred.8 <- predict(base.8$fit, newdata = testing)
pred.9 <- predict(base.9$fit, newdata = testing)
pred.10 <- predict(base.10$fit, newdata = testing)
pred.11 <- predict(base.11$fit, newdata = testing)
pred.12 <- predict(base.12$fit, newdata = testing)
pred.13 <- predict(base.13$fit, newdata = testing)
pred.14 <- predict(base.14$fit, newdata = testing)
pred.15 <- predict(base.15$fit, newdata = testing)
pred.16 <- predict(base.16$fit, newdata = testing)
pred.17 <- predict(base.17$fit, newdata = testing)
pred.18 <- predict(base.18$fit, newdata = testing)
pred.19 <- predict(base.19$fit, newdata = testing)
pred.20 <- predict(base.20$fit, newdata = testing)

all.preds <- cbind.data.frame(pred.1, pred.2, pred.3, pred.4, pred.5, 
                              pred.6, pred.7, pred.8, pred.9, pred.10,
                              pred.11, pred.12, pred.13, pred.14, pred.15,
                              pred.16, pred.17, pred.18, pred.19, pred.20)

colnames(all.preds) <- paste0("mod.", 1:ncol(all.preds))



test.features2 <- matrix(rep(0), ncol = nrow(interactions), nrow = nrow(all.preds))
for(i in 1:nrow(interactions)){
  test.features2[,i] <- all.preds[,interactions[i,1]]*all.preds[,interactions[i,2]]
}
colnames(test.features2) <- paste0("meta.Interaction.", 1:ncol(test.features2))

int.metadata.test <- cbind.data.frame(all.preds, test.features2)

test.tsne.out <- as.data.frame(predict(par.mapping, int.metadata.test))
colnames(test.tsne.out) <- paste0("meta.tsne.", 1:dim)

testing.metadata <- cbind.data.frame(int.metadata.test, test.tsne.out)

final.pred <- predict(metalearner, testing.metadata)

my.sub <- cbind.data.frame(ID = testing$Id, SalePrice = final.pred)

head(my.sub)

write.csv(my.sub, "./my_submission.csv", row.names=FALSE)

save.image("./Quantile Ensemble Prediction.RData")
