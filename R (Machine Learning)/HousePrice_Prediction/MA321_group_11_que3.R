####################################### Installing & Loading required packages ###########################################

install.packages('caret')
install.packages('randomForest')
install.packages('gbm')
install.packages('e1071')

library(caret)
library(randomForest)
library(gbm)
library(e1071)

#############################################  Loading the data ########################################################################

data <- read.csv("C:/Users/jhans/OneDrive/Documents/Spring Term/ModellingExperimentalData/MA321/house-data.csv",header=TRUE)
head(data)
str(data)

####################################### Description of column names ########################################################################
#LotFrontage: Linear feet of street connected to property
#LotArea: Lot size in square feet
#Street: Type of road access to property
#Utilities: Type of utilities available
#LotConfig: Lot configuration
#Neighborhood: Physical locations within Ames city limits
#Condition1: Proximity to various conditions
#Condition2: Proximity to various conditions (if more than one is present)
#BldgType: Type of dwelling
#HouseStyle: Style of dwelling
#OverallQual: Rates the overall material and finish of the house
#OverallCond: Rates the overall condition of the house
#YearBuilt: Original construction date
#RoofStyle: Type of roof
##RoofMatl: Roof material
#Exterior1st: Exterior covering on house
#MasVnrArea: Masonry veneer area in square feet
#ExterQual: Evaluates the quality of the material on the exterior 
#ExterCond: Evaluates the present condition of the material on the exterior
#Foundation: Type of foundation
#BsmtQual: Evaluates the height of the basement
#BsmtCond: Evaluates the general condition of the basement
#TotalBsmtSF: Total square feet of basement area
#Heating: Type of heating
#1stFlrSF: First Floor square feet
#2ndFlrSF: Second floor square feet
#LowQualFinSF: Low quality finished square feet (all floors)
#GrLivArea: Above grade (ground) living area square feet
#FullBath: Full bathrooms above grade
#Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#Kitchen: Kitchens above grade
#KitchenQual: Kitchen quality
#TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#Functional: Home functionality (Assume typical unless deductions are warranted)
#Fireplaces: Number of fireplaces
#GarageType: Garage location
#GarageArea: Size of garage in square feet
#GarageQual: Garage quality
#GarageCond: Garage condition
#PavedDrive: Paved driveway
#PoolArea: Pool area in square feet
#PoolQC: Pool quality
#Fence: Fence quality
#MiscFeature: Miscellaneous feature not covered in other categories
#MiscVal: $Value of miscellaneous feature
#MoSold: Month Sold (MM)
#YrSold: Year Sold (YYYY)
#SaleType: Type of sale
#SaleCondition: Condition of sale
#SalePrice: Price of the property
##############################################################################################################

############################# Treating missing values in numerical columns ###################################

data$LotFrontage[which(is.na(data$LotFrontage))] <- median(LotFrontage,na.rm = TRUE)
data$MasVnrArea[which(is.na(data$MasVnrArea))] <- 0

print(sum(is.na(data$MasVnrArea)))
print(sum(is.na(data$LotFrontage)))

################################ Treating missing values in categorical columns ##############################

# replacing NA's with "No alley access" in Alley column
data$Alley <- as.character(data$Alley)
data$Alley[which(is.na(data$Alley))] <- "No alley access"
data$Alley <- as.factor(data$Alley)


# replacing NA's with "No Basement" in BsmtCond, BsmtQual columns
data$BsmtCond <- as.character(data$BsmtCond)
data$BsmtCond[is.na(data$BsmtCond)] <- "No Basement"
data$BsmtCond <- as.factor(data$BsmtCond)
print(table(data$BsmtCond))

data$BsmtQual <- as.character(data$BsmtQual)
data$BsmtQual[is.na(data$BsmtQual)] <- "No Basement"
data$BsmtQual <- as.factor(data$BsmtQual)
print(table(data$BsmtQual))

# replacing NA's with ""No Garage" in GarageCond , GarageType columns
data$GarageCond <- as.character(data$GarageCond)
data$GarageCond[is.na(data$GarageCond)] <- "No Garage"
data$GarageCond <- as.factor(data$GarageCond)
print(table(data$GarageCond))

data$GarageType <- as.character(data$GarageType)
data$GarageType[is.na(data$GarageType)] <- "No Garage"
data$GarageType <- as.factor(data$GarageType)
print(table(data$GarageType))

# replacing NA's with "No Pool" in PoolQC column
data$PoolQC <- as.character(data$PoolQC)
data$PoolQC[is.na(data$PoolQC)] <- "No Pool"
data$PoolQC <- as.factor(data$PoolQC)
print(table(data$PoolQC ))

# replacing NA's with  "No Fence" in Fence column
data$Fence <- as.character(data$Fence)
data$Fence[is.na(data$Fence)] <- "No Fence"
data$Fence <- as.factor(data$Fence)
print(table(data$Fence))

# replacing NA's with "None" in Fence column
data$MiscFeature <- as.character(data$MiscFeature)
data$MiscFeature[which(is.na(data$MiscFeature))] <- "None"
data$MiscFeature <- as.factor(data$MiscFeature)
print(table(data$MiscFeature))

#################################  Factorizing the categorical columns ##############################################################
data$Alley <-  as.factor(data$Alley)
data$BsmtCond <- as.factor(data$BsmtCond)
data$BsmtQual <- as.factor(data$BsmtQual)
data$GarageCond<- as.factor(data$GarageCond)
data$GarageType<- as.factor(data$GarageType)
data$PoolQC<- as.factor(data$PoolQC)
data$Fence<- as.factor(data$Fence)
data$MiscFeature<- as.factor(data$MiscFeature)
data$Street <- as.factor(data$Street)
data$Utilities <- as.factor(data$Utilities)
data$LotConfig <- as.factor(data$LotConfig)
data$Neighborhood <- as.factor(data$Neighborhood)
data$Condition1<- as.factor(data$Condition1)
data$Condition2<- as.factor(data$Condition2)
data$BldgType<- as.factor(data$BldgType)
data$HouseStyle<- as.factor(data$HouseStyle)
data$RoofStyle<- as.factor(data$RoofStyle)
data$RoofMatl <- as.factor(data$RoofMatl)
data$Exterior1st <- as.factor(data$Exterior1st)
data$ExterQual  <- as.factor(data$ExterQual)
data$ExterCond  <- as.factor(data$ExterCond)
data$Foundation <- as.factor(data$Foundation)
data$Heating  <- as.factor(data$Heating)
data$KitchenQual <- as.factor(data$KitchenQual)
data$Functional <- as.factor(data$Functional)
data$PavedDrive <- as.factor(data$PavedDrive)
data$SaleType <- as.factor(data$SaleType)
data$SaleCondition <- as.factor(data$SaleCondition)

##################################### Remiving the 'ID' column ###############################################
data <- subset(data,select=-c(Id))
dim(data)


########################### Data partitioning for training and testing ######################################

# splitting data for training and testing
set.seed(125)
inTraining <- createDataPartition(data$SalePrice, p = .80, list = FALSE)
train_set <- data[inTraining,]
validate_set  <- data[-inTraining,]

dim(train_set)
dim(validate_set)


######################################### Model building #######################################################
#####################  Building a RandomForest Model #################################

set.seed(825)
forest_model <- randomForest(SalePrice~.,
                             data = train_set,
                             importance=TRUE)
forest_model

plot(forest_model)

### Feature Importance ###
varImpPlot(forest_model)

####################  Building a GradientBoosting Model ##############################
set.seed(825)
model_gbm <- gbm(SalePrice ~., data=train_set)
model_gbm

#####################  Building SVM model ############################################
set.seed(825)
model_svm <- svm(SalePrice ~., data=train_set)
model_svm

### prediction on validation set : Computing  RMSE and R^2 scores ###
predicted_prices_forest <- predict(forest_model, newdata=validate)
predicted_prices_gbm <- predict(model_gbm, newdata=validate)
predicted_prices_svm <- predict(model_svm, newdata=validate)

RMSE <- function(actual,predicted) {sqrt(mean((actual-predicted)^2))}
rmse_RandomForest <- RMSE(validate$SalePrice,predicted_prices_forest)
rmse_gbm <- RMSE(validate$SalePrice,predicted_prices_gbm)
rmse_svm <- RMSE(validate$SalePrice,predicted_prices_svm)

rmse_mat <- matrix(c(rmse_RandomForest,rmse_gbm,rmse_svm), nrow = 1, ncol = 3, byrow = TRUE,
               dimnames = list(c("RMSE_validation"),
                               c("  RandomForest  ", "  GBM  ", "  SVM  ")))
rmse_mat

mean_saleprice <- mean(validate$SalePrice)
R2 <- function(predicted) { 1 - (sum((validate$SalePrice-predicted)^2)/sum((validate$SalePrice-mean_saleprice)^2))}

R2_RandomForest <- R2(predicted_prices_forest)
R2_gbm <- R2(predicted_prices_gbm)
R2_svm <- R2(predicted_prices_svm)

R2_mat <- matrix(c(R2_RandomForest,R2_gbm,R2_svm), nrow = 1, ncol = 3, byrow = TRUE,
                   dimnames = list(c("R^2_validation"),
                                   c("  RandomForest  ", "  GBM  ", "  SVM  ")))
R2_mat

############################################  RandomForest ###############################################################

### k- fold cross validation  with k = 10 ###

# The function trainControl can be used to specify the type of resampling:
fitControl <- trainControl(method = "cv",
                           number = 10)

RandomForest.cv <- train(SalePrice ~. , 
                data= data,
                method = 'rf',
                trControl = fit_ctrl
                )
print(RandomForest.cv)
plot(RandomForest.cv)

### Bootstrapping with n=25 ###

RandomForest.boot <-  train(SalePrice ~. , 
                   data= data,
                   method = 'rf')
print(RandomForest.boot)
plot(RandomForest.boot)


########################################### (Gradient Boosting) #########################################################################
set.seed(825)
### k-fold crossvalidation with k=10 ###
gbmFit1_cv <- train(SalePrice~ ., 
                 data = data, 
                 method = "gbm", 
                 trControl = trainControl("cv", number = 10),
                 verbose = FALSE)
gbmFit1_cv
plot(gbmFit1_cv)
### bootstrap with n=25 ###
gbmFit1_boot <- train(SalePrice~ ., 
                      data = data, 
                      method = "gbm", 
                      verbose = FALSE)
plot(gbmFit1_boot)
gbmFit1_boot


### For a gradient boosting machine (GBM) model, there are three main tuning parameters: ###
  
# number of iterations, i.e. trees, (called n.trees in the gbm function)
# complexity of the tree, called interaction.depth
# learning rate: how quickly the algorithm adapts, called shrinkage
# the minimum number of training set samples in a node to commence splitting (n.minobsinnode)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)

set.seed(825)
gbmFit2 <- train(SalePrice ~ ., data = data, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)
gbmFit2

trellis.par.set(caretTheme())
plot(gbmFit2)  

trellis.par.set(caretTheme())
plot(gbmFit2, metric = "Rsquared")

par(mfrow=c(1,2))
ggplot(gbmFit1_cv)
ggplot(gbmFit2)

############################################# Support Vector Machine (SVM) ######################################################
### Linear kernel ###

svmFit_linear_cv <- train(SalePrice ~ ., data = data, 
                          method = "svmLinear", 
                          trControl = fitControl, # 10 fold cross validation
                          metric = "RMSE")
svmFit_linear_cv
ggplot(svmFit_linear_cv)

svmFit_linear_boot <- train(SalePrice ~ ., data = data, 
                            method = "svmLinear",    # bootstrap with n=25
                            metric = "RMSE")  
svmFit_linear_boot
ggplot(svmFit_linear_boot)

### rbf kernel ###
svmFit_rbf <- train(SalePrice ~ ., data = data, 
                    method = "svmRadial", 
                    trControl = fitControl, # 10 fold cross validation
                    metric = "RMSE")
svmFit_rbf

svmFit_rbf_boot <- train(SalePrice ~ ., data = data, 
                         method = "svmRadial", 
                         metric = "RMSE")  # bootstrap with n=25
svmFit_rbf_boot