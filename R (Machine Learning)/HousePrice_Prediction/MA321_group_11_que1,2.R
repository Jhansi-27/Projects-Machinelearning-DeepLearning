#loading required libraries
library(data.table)
library(ggplot2)
library(dplyr)
library(Amelia)
library(e1071)
library(hrbrthemes)
library(gganimate)


#loading the dataset
df <- read.csv("D:\\New folder\\Applied Statistics\\house-data.csv", header = T)
str(df)

#numerical summary statistics
summary(df)
attributes(df)

#dimensions of dataset
dim(df)

#displaying the columns having null values and number of missing values in each column
colSums(sapply(df, is.na))

#displaying the objecting of the dataset after removing feqqw columns
str(df)

#Imputatuion method for the columns having missing values i.e., LotFrontage and MasVnrArea by using median
df$LotFrontage[which(is.na(df$LotFrontage))] <- median(df$LotFrontage,na.rm = TRUE)
df$MasVnrArea[which(is.na(df$MasVnrArea))] <- 0

print(sum(is.na(df$MasVnrArea)))
print(sum(is.na(df$LotFrontage)))

################################ Treating missing values in categorical columns ##############################

# replacing NA's with "No alley access" in Alley column
df$Alley <- as.character(df$Alley)
df$Alley[which(is.na(df$Alley))] <- "No alley access"
df$Alley <- as.factor(df$Alley)


# replacing NA's with "No Basement" in BsmtCond, BsmtQual columns
df$BsmtCond <- as.character(df$BsmtCond)
df$BsmtCond[is.na(df$BsmtCond)] <- "No Basement"
df$BsmtCond <- as.factor(df$BsmtCond)
print(table(df$BsmtCond))
df$BsmtQual <- as.character(df$BsmtQual)
df$BsmtQual[is.na(df$BsmtQual)] <- "No Basement"
df$BsmtQual <- as.factor(df$BsmtQual)
print(table(df$BsmtQual))

# replacing NA's with ""No Garage" in GarageCond , GarageType columns
df$GarageCond <- as.character(df$GarageCond)
df$GarageCond[is.na(df$GarageCond)] <- "No Garage"
df$GarageCond <- as.factor(df$GarageCond)
print(table(df$GarageCond))
df$GarageType <- as.character(df$GarageType)
df$GarageType[is.na(df$GarageType)] <- "No Garage"
df$GarageType <- as.factor(df$GarageType)
print(table(df$GarageType))

# replacing NA's with "No Pool" in PoolQC column
df$PoolQC <- as.character(df$PoolQC)
df$PoolQC[is.na(df$PoolQC)] <- "No Pool"
df$PoolQC <- as.factor(df$PoolQC)
print(table(df$PoolQC ))

# replacing NA's with  "No Fence" in Fence column
df$Fence <- as.character(df$Fence)
df$Fence[is.na(df$Fence)] <- "No Fence"
df$Fence <- as.factor(df$Fence)
print(table(df$Fence))

# replacing NA's with "None" in Fence column
df$MiscFeature <- as.character(df$MiscFeature)
df$MiscFeature[which(is.na(df$MiscFeature))] <- "None"
df$MiscFeature <- as.factor(df$MiscFeature)
print(table(df$MiscFeature))

#################################  Factorizing ##############################################################
df$Alley <-  as.factor(df$Alley)
df$BsmtCond <- as.factor(df$BsmtCond)
df$BsmtQual <- as.factor(df$BsmtQual)
df$GarageCond<- as.factor(df$GarageCond)
df$GarageType<- as.factor(df$GarageType)
df$PoolQC<- as.factor(df$PoolQC)
df$Fence<- as.factor(df$Fence)
df$MiscFeature<- as.factor(df$MiscFeature)
df$Street <- as.factor(df$Street)
df$Utilities <- as.factor(df$Utilities)
df$LotConfig <- as.factor(df$LotConfig)
df$Neighborhood <- as.factor(df$Neighborhood)
df$Condition1<- as.factor(df$Condition1)
df$Condition2<- as.factor(df$Condition2)
df$BldgType<- as.factor(df$BldgType)
df$HouseStyle<- as.factor(df$HouseStyle)
df$RoofStyle<- as.factor(df$RoofStyle)
df$RoofMatl <- as.factor(df$RoofMatl)
df$Exterior1st <- as.factor(df$Exterior1st)
df$ExterQual  <- as.factor(df$ExterQual)
df$ExterCond  <- as.factor(df$ExterCond)
df$Foundation <- as.factor(df$Foundation)
df$Heating  <- as.factor(df$Heating)
df$KitchenQual <- as.factor(df$KitchenQual)
df$Functional <- as.factor(df$Functional)
df$PavedDrive <- as.factor(df$PavedDrive)
df$SaleType <- as.factor(df$SaleType)
df$SaleCondition <- as.factor(df$SaleCondition)

#Map for displaying the missing values and can be observed that there are no missing values present after imputation
colSums(sapply(df, is.na))

#correlation for the variables present in dataset which has numerical values
df_new <- df %>% select_if(is.numeric)
df_new.corr<-cor(df_new)
df_new.corr
corrplot.mixed(df_new.corr, lower.col = "black", number.cex = .6)

#creating a column "OverallCondition" in the data set to categorize the overall condition of the house w.r.t "OverallCond and categorizing as Poor, Average and Good"
table(df$OverallCond)
setDT(df)[OverallCond >1 & OverallCond <=3, OverallCondition := "Poor"]
df[OverallCond >3 & OverallCond <=6, OverallCondition := "Average"]
df[OverallCond >6 & OverallCond <=10, OverallCondition := "Good"]

#displaying the total number of times a unique value comes in the created column
df[,table(OverallCondition)]

#Plot for total rooms vs sale price and its overall condition
ggplot(df, aes(x=TotRmsAbvGrd, y=SalePrice, color=OverallCondition)) + 
  geom_point(size=6, alpha = 0.3) + theme_bw() + labs(title = "Total number of rooms vs sale price and its category", xlab = "Total rooms", ylab = "Sale Price")

#Plotting Neighborhood vs Sale Price using boxplot
ggplot(df, aes(x=Neighborhood, y=SalePrice, fill=Neighborhood)) +
  geom_boxplot() +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Neighborhood vs Saleprice") +
  xlab("Neighborhood") + ylab("Sale Price") + theme(axis.text.x = element_text(angle = 90))

#Plotting Housing market analysis using Area, Sale Price and Neighborhood
ggplot(df, aes(LotArea,GrLivArea, size = SalePrice, color = Neighborhood)) +
  geom_point() +
  scale_x_log10() +
  theme_bw() + labs(title = "Housing Market by neighborhood, Lot Area, Living Area and Sale Price", x = "Lot Area in sqft", y = "Living Area in sqft")


#converting the categorical values to numerical values for columns Neighborhood and OverallCondition
df$OverallCondition = as.factor(df$OverallCondition)
unclass(df$OverallCondition)
df$Neighborhood = as.factor(df$Neighborhood)
unclass(df$Neighborhood)

#Dividing the dataset into train data and test data with train data consisting 1000 rows and test data with 460 rows
set.seed(52)
ids <- sample(x = 1460, size = 1000, replace = F)
train <- df[ids,]
test <- df[-ids,]


#using Logistic Regression to train the model using train data and few columns adjusting according to the AIC Value
log.fit <- glm(OverallCond ~  TotRmsAbvGrd + FullBath + LotFrontage + BedroomAbvGr + MasVnrArea + TotalBsmtSF + SalePrice + MiscVal + OverallCondition + Neighborhood, data = train)

#Displaying the statistical summary of the trained model
summary(log.fit)

#Predicting the output using test data after training the model 
log.pred <- predict(log.fit, test, type = "response")
log.pred
log.pred <- round(log.pred)
log.pred <- as.numeric(log.pred)

#categorize the output predicted as Poor, Average and Good
log.class <- ifelse(log.pred >=1 & log.pred <= 3, "Poor", 
                    ifelse(log.pred >= 4 & log.pred <=6, "Average", 
                           ifelse(log.pred >=7 & log.pred<=10, "Good", NA)))

#displaying the confusion matrix
table(log.class, test$OverallCondition)


#Using Naivebayes to train the model
Bayes.fit <- naiveBayes(OverallCond ~  TotRmsAbvGrd + FullBath + LotFrontage + BedroomAbvGr + MasVnrArea + TotalBsmtSF + SalePrice + MiscVal + OverallCondition + Neighborhood, data = train)

#Predicting the output for test data after completing the training 
testPred=predict(Bayes.fit, newdata=test, type="class")
testPred <- as.numeric(testPred)

#categorize the output predicted as Poor, Average and Good
testclass <- ifelse(testPred >=1 & testPred <= 3, "Poor", 
                    ifelse(testPred >= 4 & testPred <=6, "Average", 
                           ifelse(testPred >=7 & testPred<=10, "Good", NA)))

#creating a table for predicted output and original values in test data
testTable=table(test$OverallCondition, testclass)
testTable

#Calculating the accuracy of the trained model
testAcc=(testTable[1,1]+testTable[2,2]+testTable[3,3])/sum(testTable)
testAcc



