
#load package
library(HSAUR2)   
library(ISLR)
library(xtable) 
library(randomForest)
############# LOADING DATA ################
data <- read.csv('"C:/Users/jhans/OneDrive/Documents/Spring Term/ModellingExperimentalData/MA321/house-data.csv",header=TRUE)




############# IMPUTING MISSING VALUES ###############

data$LotFrontage[is.na(data$LotFrontage)] <- median(data$LotFrontage,na.rm = TRUE)


data$MasVnrArea[is.na(data$MasVnrArea)] <- 0


data$Alley[is.na(data$Alley)] <- "No alley access"


data$BsmtCond[is.na(data$BsmtCond)] <- "No Basement"
data$BsmtQual[is.na(data$BsmtQual)] <- "No Basement"

data$GarageCond[is.na(data$GarageCond)] <- "No Garage"
data$GarageType[is.na(data$GarageType)] <- "No Garage"

data$PoolQC[is.na(data$PoolQC)] <- "No Pool"
data$Fence[is.na(data$Fence)] <- "No Fence"
data$MiscFeature[is.na(data$MiscFeature)] <- "None"


############### CONVERTING TO FACTORS ##################
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



################ CONVERTING TO NUMERIc  ##########################

data$Street <- as.numeric(data$Street)

data$Alley <- as.numeric(data$Alley)

data$Fence <- as.numeric(data$Fence)

data$Utilities <- as.numeric(data$Utilities)

data$LotConfig <- as.numeric(data$LotConfig)

data$Neighborhood <- as.numeric(data$Neighborhood)

data$Condition1 <- as.numeric(data$Condition1)

data$Condition2 <- as.numeric(data$Condition2)

data$BldgType <- as.numeric(data$BldgType)

data$HouseStyle <- as.numeric(data$HouseStyle)

data$RoofStyle <- as.numeric(data$RoofStyle)

data$RoofMatl <- as.numeric(data$RoofMatl)

data$Exterior1st <- as.numeric(data$Exterior1st)

data$ExterQual <- as.numeric(data$ExterQual)

data$ExterCond <- as.numeric(data$ExterCond)

data$Foundation <- as.numeric(data$Foundation)

data$BsmtQual <- as.numeric(data$BsmtQual)

data$BsmtCond <- as.numeric(data$BsmtCond)

data$Heating <- as.numeric(data$Heating)

data$KitchenQual <- as.numeric(data$KitchenQual)

data$Functional <- as.numeric(data$Functional)

data$GarageType <- as.numeric(data$GarageType)

data$GarageCond <- as.numeric(data$GarageCond)

data$PavedDrive <- as.numeric(data$Neighborhood)

data$PoolQC <- as.numeric(data$PoolQC)

data$MiscFeature <- as.numeric(data$MiscFeature)

data$SaleType <- as.numeric(data$SaleType)

data$SaleCondition <- as.numeric(data$SaleCondition)




########################## REMOVING ID COLUMN ###################################


data <- subset(data,select=-c(Id))
dim(data)

####################################### SPLITTING DATA ###################################
set.seed(1)
samp <- sample(nrow(data), nrow(data)*0.75)
house.train <- data[samp,]
house.valid <- data[-samp,]


############################################# PCA #####################################
prin_comp <- prcomp(house.train, scale = T)


#scree plot
plot(prin_comp, xlab = "Principal Component",
             ylab = "Proportion of Variance Explained",
             type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
              ylab = "Cumulative Proportion of Variance Explained",
              type = "b")


############################### PRINCIPLE COMPONENTS FOR TRAINING DATA ##################################
train.data <- data.frame(SalePrice = house.train$SalePrice, prin_comp$x)

#we are interested in first 35 PCAs
train.data <- train.data[,1:35]




##################################  BUILDING A RANDOM FOREST MODEL #######################################
Rand_model_PCA <- randomForest(SalePrice~.,
                             data = train.data,
                             importance=TRUE)
Rand_model_PCA

plot(Rand_model_PCA)

##############################PRINCIPLE COMPONENTS FOR VALIDATION DATA ###########################

test.data <- predict(prin_comp,newdata=house.valid)

# SELECTING FIRST 35 PCA's
test.data <- test.data[,1:35]

############################# PREDICTING ON VALIDATION SET #################################
Predicted_Sale_Prices_forest_PCA <- predict(Rand_model_PCA, newdata=test.data)



################ CALCULATING R2 SCORE####################################
mean_sale_price_house <- mean(house.valid$SalePrice)
R2_score <- function(predicted) { 1 - (sum((house.valid$SalePrice-predicted)^2)/sum((house.valid$SalePrice-mean_sale_price_house)^2))}

R2_Score_Rand_Forest <- R2_score(Predicted_Sale_Prices_forest_PCA)
R2_Score_Rand_Forest


