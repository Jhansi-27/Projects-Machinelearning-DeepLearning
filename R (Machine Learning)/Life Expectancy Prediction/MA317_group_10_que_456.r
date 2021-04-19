####################################################################################
#Dataset columns
#SP.DYN.LE00.IN Life expectancy at birth, total (years)
#EG.ELC.ACCS.ZS Access to electricity (\% of population)
#NY.ADJ.NNTY.KD.ZG Adjusted net national income (annual \% growth)
#NY.ADJ.NNTY.KD Adjusted net national income (constant 2010 US\$)
#SE.PRM.UNER.ZS Children out of school (\% of primary school age)
#SE.XPD.PRIM.ZS Expenditure on primary education (\% of government expenditure on education)
#SP.DYN.IMRT.IN Mortality rate, infant (per 1,000 live births)
#SE.ADT.LITR.ZS Literacy rate, adult total (\% of people ages 15 and above)
#SP.POP.GROW Population growth (annual \%)
#SP.POP.TOTL Population, total
#SE.PRM.CMPT.ZS Primary completion rate, total (\% of relevant age group)
#SH.XPD.CHEX.GD.ZS Current health expenditure (\% of GDP)
#SH.XPD.CHEX.PP.CD Current health expenditure per capita, PPP (current international \$)
#SL.UEM.TOTL.NE.ZS Unemployment, total (\% of total labor force) (national estimate)
#SP.DYN.AMRT.FE Mortality rate, adult, female (per 1,000 female adults)
#SP.DYN.AMRT.MA Mortality rate, adult, male (per 1,000 male adults)
#NY.GDP.MKTP.KD.ZG GDP growth (annual \%)
#NY.GDP.PCAP.PP.CD GDP per capita, PPP (current international \$)
#SP.DYN.CBRT.IN Birth rate, crude (per 1,000 people)
#NY.GNP.PCAP.PP.CD GNI per capita, PPP (current international \$)
#SL.EMP.TOTL.SP.ZS Employment to population ratio, 15+, total (\%) (modeled ILO estimate)

################################ Load the libraries##################################
library(ggplot2)
library(tidyverse)
library(ggthemes) 
library(scales)
library(MASS)
library(olsrr)
library(leaps)
library(mice)
library(car)
library(naniar)
############################### Load the Dataset ####################################

#load the data into a dataframe
data <- read.csv("C:/Users/jhans/OneDrive/Documents/Spring Term/ModellingExperimentalData/MA317/LifeExpectancyData1.csv",header=TRUE)

#check few rows
head(data)

#check the datatype of columns and few values
print(str(data))

## Observations : 
#The life expentancy dataset has 232 observations(or rows) and 23 features with 'SP.DYN.LE00.IN' as target feature.

# attaching the dataset to the environment
attach(data)

####################################### renaming all columns #############################################
data <- data %>% 
    rename('life_expectancy_at_birth'='SP.DYN.LE00.IN',
            'per_pop_access_to_electricity'='EG.ELC.ACCS.ZS',
            'per_annual_growth_national_income'='NY.ADJ.NNTY.KD.ZG',
            'net_national_income'='NY.ADJ.NNTY.KD',
            'per_children_out_of_school'='SE.PRM.UNER.ZS',
            'per_expenditure_primary_education'='SE.XPD.PRIM.ZS',
            'infant_mortality_rate_per_1000'='SP.DYN.IMRT.IN',
            'per_adult_literacy_rate'='SE.ADT.LITR.ZS',
            'per_annual_population_growth'='SP.POP.GROW',
            'total_population'='SP.POP.TOTL',
            'per_primary_completion'='SE.PRM.CMPT.ZS',
            'per_gdp_health_expenditure'='SH.XPD.CHEX.GD.ZS',
            'health_expenditure_per_capita'='SH.XPD.CHEX.PC.CD',
            'per_unemployment'='SL.UEM.TOTL.NE.ZS',
            'per_adult_female_mortality_rate_per_1000'='SP.DYN.AMRT.FE',
            'per_adult_male_mortality_rate_per_1000'='SP.DYN.AMRT.MA',
            'per_annual_gdp_growth'='NY.GDP.MKTP.KD.ZG',
            'gdp_per_capita'='NY.GDP.PCAP.PP.CD',
            'crude_birth_rate_per_1000'='SP.DYN.CBRT.IN',
            'gni_per_capita'='NY.GNP.PCAP.PP.CD',
            'employment_to_population_ratio'='SL.EMP.TOTL.SP.ZS'

            ) 

head(data)

##################################### Data Analysis ####################################

# Counting the number of columns that consists of numerical data
num_cols <- unlist(lapply(data, is.numeric))         
cat("Total number of numeric columns: ", ncol(data[num_cols]))

#statistics of target variable
summary(data$life_expectancy_at_birth)

#plotting histogram for the target variable
hist(data$life_expectancy_at_birth)

# looking at some basic statistics
data %>% 
      summarize(count = n(),
             avg_life_expectancy_at_birth = mean(life_expectancy_at_birth, na.rm=TRUE),
             avg_national_income = mean(net_national_income, na.rm=TRUE),
             avg_infant_mortality_rate_per_1000 = mean(infant_mortality_rate_per_1000, na.rm=TRUE),
            avg_gdp_per_capita = mean(gdp_per_capita, na.rm=TRUE))


# Null value imputation done by others
vis_miss(data)
gg_miss_var(data)+ labs(y = "Missingness of values in variables")

#Removing the columns not needed for training
data <- subset(data,select=-c(Country.Name,Country.Code))

############################################ Missing value imputation#############################

## Finding the number of missing values in each column of dataset
colSums(sapply(data, is.na))

#Missing value imputation using mean median
data$life_expectancy_at_birth[is.na(data$life_expectancy_at_birth)] <- mean(data$life_expectancy_at_birth,na.rm = TRUE)
data$per_annual_growth_national_income[is.na(data$per_annual_growth_national_income)] <- mean(data$per_annual_growth_national_income,na.rm = TRUE)
data$per_children_out_of_school[is.na(data$per_children_out_of_school)] <- median(data$per_children_out_of_school,na.rm = TRUE)
data$per_expenditure_primary_education [is.na(data$per_expenditure_primary_education )] <- mean(data$per_expenditure_primary_education ,na.rm = TRUE)
data$infant_mortality_rate_per_1000[is.na(data$infant_mortality_rate_per_1000)] <- mean(data$infant_mortality_rate_per_1000,na.rm = TRUE)
data$per_adult_literacy_rate[is.na(data$per_adult_literacy_rate)] <- mean(data$per_adult_literacy_rate,na.rm = TRUE)
data$per_annual_population_growth[is.na(data$per_annual_population_growth)] <- median(data$per_annual_population_growth,na.rm = TRUE)
data$total_population[is.na(data$total_population)] <- median(data$total_population,na.rm = TRUE)
data$per_primary_completion[is.na(data$per_primary_completion)] <- mean(data$per_primary_completion,na.rm = TRUE)
data$per_gdp_health_expenditure[is.na(data$per_gdp_health_expenditure)] <- mean(data$per_gdp_health_expenditure,na.rm = TRUE)
data$health_expenditure_per_capita[is.na(data$health_expenditure_per_capita)] <- mean(data$health_expenditure_per_capita,na.rm = TRUE)
data$per_unemployment[is.na(data$per_unemployment)] <- median(data$per_unemployment,na.rm = TRUE)
data$per_adult_female_mortality_rate_per_1000[is.na(data$per_adult_female_mortality_rate_per_1000)] <- mean(data$per_adult_female_mortality_rate_per_1000,na.rm = TRUE)
data$per_adult_male_mortality_rate_per_1000[is.na(data$per_adult_male_mortality_rate_per_1000)] <- mean(data$per_adult_male_mortality_rate_per_1000,na.rm = TRUE)
data$per_annual_gdp_growth[is.na(data$per_annual_gdp_growth)] <- mean(data$per_annual_gdp_growth,na.rm = TRUE)
data$gdp_per_capita[is.na(data$gdp_per_capita)] <- median(data$gdp_per_capita,na.rm = TRUE)
data$gni_per_capita[is.na(data$gni_per_capita)] <- mean(data$gni_per_capita,na.rm = TRUE)
data$net_national_income[is.na(data$net_national_income )] <- mean(data$net_national_income ,na.rm = TRUE)
data$employment_to_population_ratio[is.na(data$employment_to_population_ratio)] <- mean(data$employment_to_population_ratio,na.rm = TRUE)

# checking if the imputation is done for all the columns
colSums(sapply(data, is.na))

##############################################################Model Building ######################################
#1. Full Model
full_model <- lm(formula=life_expectancy_at_birth ~.,data=data)
summary(full_model)

#2. Reduced Model using backward selection method
#Perform Backward selection
stepAIC(full_model, direction = "backward", trace = FALSE)

selected_model = lm(formula = life_expectancy_at_birth ~ per_annual_growth_national_income   + 
                      infant_mortality_rate_per_1000 + per_gdp_health_expenditure   + 
                      health_expenditure_per_capita   + per_adult_female_mortality_rate_per_1000   + 
                      per_adult_male_mortality_rate_per_1000 + crude_birth_rate_per_1000   + gni_per_capita  , 
                    data = data)

summary(selected_model)

#3. Reduced Model using leaps (Mallows' Cp value)

full_model.cp <- lm(formula=life_expectancy_at_birth ~.,data=data,x=TRUE)

X <- full_model.cp$x
y <-data$life_expectancy_at_birth

all.models <- leaps(X, y, int = FALSE, strictly.compatible = FALSE, method="Cp")

#plot the cp value for each model
plot(all.models$size, all.models$Cp, log="y", xlab="|M|", ylab=expression(C[p]),ylim=c(1,200))
lines(all.models$size, all.models$size) # this plots the line C_p=|M|

#find the model with lowest cp value 
min.cp <- all.models$Cp == min(all.models$Cp) #this finds the smallest C_p value
min.cp <- all.models$which[min.cp, ] #this finds the corresponding model with the smallest C_p
min.cp #

selected_model_cp = lm(formula = life_expectancy_at_birth ~ per_annual_growth_national_income + 
                         infant_mortality_rate_per_1000 + health_expenditure_per_capita + 
                         per_gdp_health_expenditure + per_adult_female_mortality_rate_per_1000 + 
                         per_adult_male_mortality_rate_per_1000 + crude_birth_rate_per_1000 +
                         gni_per_capita, 
                       data = data)
summary(selected_model_cp)

#4. Model using the vif score 

# checking the vif score
vif(full_model)

# selected the columns having vif score less than 5
selected_model_vif = lm(formula = life_expectancy_at_birth ~ per_annual_growth_national_income + 
                          net_national_income + per_children_out_of_school + 
                          per_expenditure_primary_education + per_adult_literacy_rate + 
                          per_annual_population_growth + total_population +
                          per_primary_completion + per_gdp_health_expenditure +
                          health_expenditure_per_capita + per_unemployment +
                          per_annual_gdp_growth + employment_to_population_ratio +
                          crude_birth_rate_per_1000 + per_pop_access_to_electricity +
                          infant_mortality_rate_per_1000 + crude_birth_rate_per_1000 +
                          per_adult_male_mortality_rate_per_1000 + per_adult_female_mortality_rate_per_1000,
                        data = data)
summary(selected_model_vif)

# Both the Models 2 and 3 have same features 

########################################
## Plotting 

#plots the standardised residuals against fitted values for FULL model
stdres_fullmodel<-rstandard(full_model)
plot(full_model$fitted.values,stdres_fullmodel,pch=16,
     ylab="Standardized Residuals",xlab="fitted y",ylim=c(-3,3),main="Full model")
abline(h=0)
abline(h=2,lty=2)
abline(h=-2,lty=2)

#plots the standardised residuals against fitted values for selected model
stdres_selected_model<-rstandard(selected_model)
plot(selected_model$fitted.values,stdres_selected_model,pch=16,
     ylab="Standardized Residuals",xlab="fitted y",ylim=c(-3,3),main="selected model(Backward Selection)")
abline(h=0)
abline(h=2,lty=2)
abline(h=-2,lty=2)

#plots the QQ-plot for the FULL model
qqnorm(stdres_fullmodel, ylab="Standardized Residuals",
       xlab="Normal Scores", main="QQ Plot for Full model" )
qqline(stdres_fullmodel)

#plots the QQ-plot for the selected model
qqnorm(stdres_selected_model, ylab="Standardized Residuals",
       xlab="Normal Scores", main="QQ Plot for Selected model(Backward Selection)" )
qqline(stdres_selected_model)


######################################
#Model Evaluation

### In order for us to chose between the reduced and the full model we need to use one of the model selection criteria

# Adjusted r-sqare value
print(paste0("adj. r-square value of Full Model is : ",summary(full_model)$adj.r.squared))
print(paste0("adj. r-square value of selected Model is : ",summary(selected_model)$adj.r.squared))
print(paste0("adj. r-square value of selected Model - VIF is : ",summary(selected_model_vif)$adj.r.squared))

### We will consider calculating the AIC and Mallow's Cp. The AIC can be easily found using the command 'AIC(. . . )'
print(paste0("AIC Score of Full Model is : ",AIC(full_model)))
print(paste0("AIC Score of selected Model-BS is : ",AIC(selected_model)))
print(paste0("AIC Score of selected Model-VIF is : ",AIC(selected_model_vif)))

#The better fitting model is the one with the lowest value, 
#which in this case is the Selected model with AIC = 971.7 compared with Full Model with  AIC = 989.7

### We can determine if the reduced model is a viable fit by calculating Mallow's Cp, 
#using the 'ols_mallows_cp' function from the 'olsrr' package. 
#To use this function we enter the reduced model first, followed by the full model

ols_mallows_cp(selected_model,full_model)  
ols_mallows_cp(selected_model_vif,full_model)  
# As this value is much less than the numbe rof features (8). Our Selected model is an acceptable model

###################################### prediction ############################################

# load the testdata into a dataframe
test_data <- read.csv("LifeExpectancyData2.csv",header=TRUE)

# check the dimension of test dataset
dim(test_data)

test_data <- test_data %>% 
  rename( 'per_pop_access_to_electricity'='EG.ELC.ACCS.ZS',
         'per_annual_growth_national_income'='NY.ADJ.NNTY.KD.ZG',
         'net_national_income'='NY.ADJ.NNTY.KD',
         'per_children_out_of_school'='SE.PRM.UNER.ZS',
         'per_expenditure_primary_education'='SE.XPD.PRIM.ZS',
         'infant_mortality_rate_per_1000'='SP.DYN.IMRT.IN',
         'per_adult_literacy_rate'='SE.ADT.LITR.ZS',
         'per_annual_population_growth'='SP.POP.GROW',
         'total_population'='SP.POP.TOTL',
         'per_primary_completion'='SE.PRM.CMPT.ZS',
         'per_gdp_health_expenditure'='SH.XPD.CHEX.GD.ZS',
         'health_expenditure_per_capita'='SH.XPD.CHEX.PC.CD',
         'per_unemployment'='SL.UEM.TOTL.NE.ZS',
         'per_adult_female_mortality_rate_per_1000'='SP.DYN.AMRT.FE',
         'per_adult_male_mortality_rate_per_1000'='SP.DYN.AMRT.MA',
         'per_annual_gdp_growth'='NY.GDP.MKTP.KD.ZG',
         'gdp_per_capita'='NY.GDP.PCAP.PP.CD',
         'crude_birth_rate_per_1000'='SP.DYN.CBRT.IN',
         'gni_per_capita'='NY.GNP.PCAP.PP.CD',
         'employment_to_population_ratio'='SL.EMP.TOTL.SP.ZS'
  ) 

# Dataframe to write the prediction into
Test_final_data <- subset(test_data,select=c(Country.Name,Country.Code))

#preprocessing as in the training dataset
test_data <- subset(test_data,select=-c(Country.Name,Country.Code))

# checking if the imputation is done for all the columns
colSums(sapply(test_data, is.na))

#Missing value imputation using mean median of training Dataset
test_data$per_annual_growth_national_income[is.na(test_data$per_annual_growth_national_income)] <- mean(data$per_annual_growth_national_income,na.rm = TRUE)
test_data$per_children_out_of_school[is.na(test_data$per_children_out_of_school)] <- median(data$per_children_out_of_school,na.rm = TRUE)
test_data$per_expenditure_primary_education [is.na(test_data$per_expenditure_primary_education )] <- mean(data$per_expenditure_primary_education ,na.rm = TRUE)
test_data$infant_mortality_rate_per_1000[is.na(test_data$infant_mortality_rate_per_1000)] <- mean(data$infant_mortality_rate_per_1000,na.rm = TRUE)
test_data$per_adult_literacy_rate[is.na(test_data$per_adult_literacy_rate)] <- mean(data$per_adult_literacy_rate,na.rm = TRUE)
test_data$per_annual_population_growth[is.na(test_data$per_annual_population_growth)] <- median(data$per_annual_population_growth,na.rm = TRUE)
test_data$total_population[is.na(test_data$total_population)] <- median(data$total_population,na.rm = TRUE)
test_data$per_primary_completion[is.na(test_data$per_primary_completion)] <- mean(data$per_primary_completion,na.rm = TRUE)
test_data$per_gdp_health_expenditure[is.na(test_data$per_gdp_health_expenditure)] <- mean(data$per_gdp_health_expenditure,na.rm = TRUE)
test_data$health_expenditure_per_capita[is.na(test_data$health_expenditure_per_capita)] <- mean(data$health_expenditure_per_capita,na.rm = TRUE)
test_data$per_unemployment[is.na(test_data$per_unemployment)] <- median(data$per_unemployment,na.rm = TRUE)
test_data$per_adult_female_mortality_rate_per_1000[is.na(test_data$per_adult_female_mortality_rate_per_1000)] <- mean(data$per_adult_female_mortality_rate_per_1000,na.rm = TRUE)
test_data$per_adult_male_mortality_rate_per_1000[is.na(test_data$per_adult_male_mortality_rate_per_1000)] <- mean(data$per_adult_male_mortality_rate_per_1000,na.rm = TRUE)
test_data$per_annual_gdp_growth[is.na(test_data$per_annual_gdp_growth)] <- mean(data$per_annual_gdp_growth,na.rm = TRUE)
test_data$gdp_per_capita[is.na(test_data$gdp_per_capita)] <- median(data$gdp_per_capita,na.rm = TRUE)
test_data$gni_per_capita[is.na(test_data$gni_per_capita)] <- mean(data$gni_per_capita,na.rm = TRUE)
test_data$net_national_income[is.na(test_data$net_national_income )] <- mean(data$net_national_income ,na.rm = TRUE)
test_data$employment_to_population_ratio[is.na(test_data$employment_to_population_ratio)] <- mean(data$employment_to_population_ratio,na.rm = TRUE)

# checking if the imputation is done for all the columns
colSums(sapply(test_data, is.na))

# prediction of value
y_predicted <- predict(selected_model, test_data)
y_predicted
#store the details into the dataframe having country name and code
Test_final_data <- cbind(Test_final_data,y_predicted)

#write to a csv file
write.csv(Test_final_data,'Predicted_life_expectancy.csv')

########################################################################################################
# One Way Anova

# library to group countries into group.
data <- read.csv("C:/Users/smrut/Desktop/Study/University_of_Essex/Modelling_experimental_Data_MA317/Assignment/LifeExpectancyData1.csv",header=TRUE)

library(countrycode)

names(data)
# Grouping countries into continents
data['continent'] <- countrycode(sourcevar = data[,"Country.Name"],
                                 origin = "country.name",
                                 destination = "continent")
# Conducting One Way ANOVA
anova1way<-aov(data$life_expectancy_at_birth~as.factor(continent),data=data)
# Summary of the ANOVA model
summary(anova1way)

# Plotting Differences in mean levels of life expectancy across continents
tukey.diet<-TukeyHSD(anova1way)
plot(tukey.diet)
#+labs(x = "Differences in mean levels of life expectancy across continents")

