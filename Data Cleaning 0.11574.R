


# Libraries and Directory -------------------------------------------------

library(Amelia)
library(mice)
library(caret)
library(lattice)

setwd("H:/tklarkin_backup/Documents/Kaggle/Housing")

# Loading Data ------------------------------------------------------------

train.raw <- read.csv("./Data/train.csv")
str(train.raw)

# train.raw$YearBuilt <- as.factor(train.raw$YearBuilt)
# train.raw$YearRemodAdd <- as.factor(train.raw$YearRemodAdd)
# train.raw$GarageYrBlt <- as.factor(train.raw$GarageYrBlt)
train.raw$MSSubClass <- as.factor(train.raw$MSSubClass)
train.raw$MoSold <- as.factor(train.raw$MoSold)
train.raw$YrSold <- as.factor(train.raw$YrSold)

test.raw <- read.csv("./Data/test.csv")
str(test.raw)

# test.raw$YearBuilt <- as.factor(test.raw$YearBuilt)
# test.raw$YearRemodAdd <- as.factor(test.raw$YearRemodAdd)
# test.raw$GarageYrBlt <- as.factor(test.raw$GarageYrBlt)
test.raw$MSSubClass <- as.factor(test.raw$MSSubClass)
test.raw$MoSold <- as.factor(test.raw$MoSold)
test.raw$YrSold <- as.factor(test.raw$YrSold)


# Plotting Missing --------------------------------------------------------

missmap(train.raw[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)

missmap(test.raw[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)

# Number Missing ----------------------------------------------------------

# Let's also get some hard numbers
sort(sapply(train.raw, function(x) { sum(is.na(x)) }), decreasing=TRUE)

sort(sapply(test.raw, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#These have too many missing or two few levels
exclude <- c('PoolQC', 'MiscFeature', 'Alley', 'Fence', "Utilities")
include.train <- setdiff(names(train.raw), exclude)
include.test <- setdiff(names(test.raw), exclude)

train.raw <- train.raw[include.train]
test.raw <- test.raw[include.test] #removing sales price

# Imputing using MICE and RF ----------------------------------------------

imp.train.raw <- mice(train.raw, m=5, method='rf', seed = 123)
imp.test.raw <- mice(test.raw, m=5, method='rf', seed = 123)

# Check Plots -------------------------------------------------------------

xyplot(imp.train.raw, LotFrontage ~ LotArea)
densityplot(imp.train.raw, ~LotFrontage)

table(train.raw$GarageType)
table(imp.train.raw$imp$GarageType)

table(train.raw$GarageFinish)
table(imp.train.raw$imp$GarageFinish)

table(train.raw$BsmtExposure)
table(imp.train.raw$imp$BsmtExposure)


# Creating New Training and Test ------------------------------------------

train.complete <- complete(imp.train.raw)
#Confirm no NAs
sum(sapply(train.complete, function(x) { sum(is.na(x)) }))

test.complete <- complete(imp.test.raw)
#Confirm no NAs
sum(sapply(test.complete, function(x) { sum(is.na(x)) }))

# Creating Dummies --------------------------------------------------------

dum.training <- dummyVars(" ~ .", data = train.complete)
dum.training <- as.data.frame(predict(dum.training, newdata = train.complete))

dum.testing <- dummyVars(" ~ .", data = test.complete)
dum.testing <- as.data.frame(predict(dum.testing, newdata = test.complete))

#Making sure same variables are in training and testing
dum.training <- dum.training[,c(intersect(names(dum.training), 
                                                  names(dum.testing)), "SalePrice")]
dum.testing <- dum.testing[,intersect(names(dum.training), names(dum.testing))]


# Adding Interactions with EBglmnet ---------------------------------------

library(EBglmnet)

eben <- cv.EBglmnet(x = as.matrix(dum.training[,-c(1,ncol(dum.training))]), 
                       y = dum.training$SalePrice, prior = "elastic net", Epis = TRUE)

eblasso <- cv.EBglmnet(x = as.matrix(dum.training[,-c(1,ncol(dum.training))]), 
                    y = dum.training$SalePrice, prior = "lassoNEG", Epis = TRUE)

#List of interactions to include
interactions <- rbind(eben$fit[,1:2], eblasso$fit[,1:2])
interactions <- interactions[which(interactions[,1]-interactions[,2] != 0),]

#
train.features2 <- matrix(rep(0), ncol = nrow(interactions), nrow = nrow(dum.training))
for(i in 1:nrow(interactions)){
  train.features2[,i] <- dum.training[,interactions[i,1]]*dum.training[,interactions[i,2]]
}

test.features2 <- matrix(rep(0), ncol = nrow(interactions), nrow = nrow(dum.testing))
for(i in 1:nrow(interactions)){
  test.features2[,i] <- dum.testing[,interactions[i,1]]*dum.testing[,interactions[i,2]]
}


colnames(train.features2) <- paste0("Interaction.", 1:ncol(train.features2))
colnames(test.features2) <- paste0("Interaction.", 1:ncol(test.features2))

int.dum.training <- cbind.data.frame(dum.training, train.features2)
int.dum.testing <- cbind.data.frame(dum.testing, test.features2)


# Removing Near-Zero Features ---------------------------------------------

#Removing near zero vars
int.dum.training <- int.dum.training[,-nearZeroVar(int.dum.training, freqCut = 99/1)]

#Making sure same variables are in training and testing
int.dum.training <- int.dum.training[,c(intersect(names(int.dum.training), 
                                                  names(int.dum.testing)), "SalePrice")]
int.dum.testing <- int.dum.testing[,intersect(names(int.dum.training), names(int.dum.testing))]

# Adding Deep Learning Features -------------------------------------------

# library(h2o)
# h2o.init()
# deep.training <- as.h2o(cbind.data.frame(
#   subset(int.dum.training, select = -c(Id, SalePrice)), SalePrice = int.dum.training$SalePrice))
# deep.testing <- as.h2o(subset(int.dum.testing, select = -c(Id)))
# 
# dnn = h2o.deeplearning(x = 1:(ncol(deep.training)-1), y = ncol(deep.training),
#                        training_frame = deep.training, hidden = c(5, 5))
# train.deepfeatures_layer1 = as.data.frame(h2o.deepfeatures(dnn, deep.training, layer = 1))
# train.deepfeatures_layer2 = as.data.frame(h2o.deepfeatures(dnn, deep.training, layer = 2))
# 
# test.deepfeatures_layer1 = as.data.frame(h2o.deepfeatures(dnn, deep.testing, layer = 1))
# test.deepfeatures_layer2 = as.data.frame(h2o.deepfeatures(dnn, deep.testing, layer = 2))
# 
# training.dnn <- cbind.data.frame(train.deepfeatures_layer1, train.deepfeatures_layer2)
# testing.dnn <- cbind.data.frame(test.deepfeatures_layer1, test.deepfeatures_layer2)
# 
# deep.int.dum.training <- cbind.data.frame(int.dum.training, training.dnn)
# deep.int.dum.testing <- cbind.data.frame(int.dum.testing, testing.dnn)

# Adding t-SNE ------------------------------------------------------------

library(Rtsne)
library(earth)

dim <- 4

tsne.data <- subset(int.dum.training, select = -c(Id, SalePrice))

set.seed(123)
train.tsne.out <- as.data.frame(Rtsne(X = tsne.data,
                                      dim = dim, theta = 0, perplexity = 50)$Y)
colnames(train.tsne.out) <- paste0("tsne.", 1:dim)

#Making parametric mapping with MARS
par.mapping <- earth(y = train.tsne.out, x = tsne.data ,
                     degree = 3, pmethod = "cv", nfold = 10)

test.tsne.out <- as.data.frame(predict(par.mapping, int.dum.testing))
colnames(test.tsne.out) <- paste0("tsne.", 1:dim)


# Writing Data Out --------------------------------------------------------

training <- cbind.data.frame(int.dum.training, train.tsne.out)
testing <- cbind.data.frame(int.dum.testing, test.tsne.out)

write.csv(training, "./training.csv", row.names=FALSE)
write.csv(testing, "./testing.csv", row.names=FALSE)

# # Feature Importance Random Forest ----------------------------------------

library(party)

set.seed(123)
model.rf <- cforest(SalePrice ~ ., data = training[,-1], 
                    controls = cforest_unbiased(ntree = 5000))

imps <- data.frame(Features = varimp(model.rf))

imps[order(imps$Features, decreasing = TRUE), , drop = FALSE]

save.image("./Data Cleaning.RData")
