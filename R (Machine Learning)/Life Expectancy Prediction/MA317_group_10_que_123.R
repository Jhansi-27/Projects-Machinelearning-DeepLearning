#Loading libraries
install.packages('ggplot2')
install.packages('countrycode')
install.packages('dplyr') 
install.packages('car')
install.packages('corrplot')


library(ggplot2)
library(countrycode)
library(dplyr)
library(car)
library(corrplot)


#loading the data
df <- read.csv("C:/Users/jhans/OneDrive/Documents/Spring Term/ModellingExperimentalData/MA317/LifeExpectancyData1.csv",header=TRUE)
str(df)

#numerical summary statistics
summary(df)
attributes(df)

#renaming the column names of dataset
df <- df %>% 
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


#Creating a column with name continent and categorizing the country according to continent using countrycode()
df$Continent = countrycode(sourcevar = df[, "Country.Name"],
                           origin = "country.name",
                           destination = "region")

#Plotting GDP per capita vs Life Expectancy with population by Continents
ggplot(df, aes(x = gdp_per_capita, y = life_expectancy_at_birth, size = total_population, color = Continent)) +
  geom_point() +
  scale_x_log10() +
  theme_bw() +labs( x = 'GDP per capita', y = 'life expectancy')

#Plotting Employment to population ratio vs Unemployment
ggplot(df, aes(x=employment_to_population_ratio, y=per_unemployment, size = total_population)) +
  geom_point(alpha=0.7) + labs(title = "Employment ratio vs Unemployment", xlab = "Employment Ratio", ylab = "Unemployment")


#Plotting Literacy rate vs Unemployment
Literacy_rate_adult_total = df[,10]
Unemployment_total = df[,16]
plot(Literacy_rate_adult_total, Unemployment_total, pch=16, main = "Literacy rate vs Unemployment")

#Plotting Male Mortatlity rate according to continent
ggplot(df, aes(x=Continent, y=per_adult_male_mortality_rate_per_1000, fill = Continent)) +
  geom_boxplot() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Male mortality rate by country") +
  xlab("Country") + ylab("Male Mortality rate(per 1000 adults)") +   theme(axis.text.x = element_text(angle = 90))


#Plotting GDP vs GNI per capita
ggplot(df, aes(x=gdp_per_capita, y=gni_per_capita)) +
  geom_point() + 
  geom_segment( aes(x=gdp_per_capita, xend=gdp_per_capita, y=0, yend=gni_per_capita))


#Plotting Children out of school vs Expenditure on Primary Educaation
Children_out_of_school = df[,7]
Expenditure_on_primary_education = df[,8]
plot(Children_out_of_school, Expenditure_on_primary_education, pch=16)


#Removing the columns not needed for training
df <- subset(df,select=-c(Country.Name,Country.Code,Continent))
colSums(sapply(df, is.na))


#Missing value imputation using mean median
df$life_expectancy_at_birth[is.na(df$life_expectancy_at_birth)] <- mean(df$life_expectancy_at_birth,na.rm = TRUE)
df$per_annual_growth_national_income[is.na(df$per_annual_growth_national_income)] <- mean(df$per_annual_growth_national_income,na.rm = TRUE)
df$per_children_out_of_school[is.na(df$per_children_out_of_school)] <- median(df$per_children_out_of_school,na.rm = TRUE)
df$per_expenditure_primary_education [is.na(df$per_expenditure_primary_education )] <- mean(df$per_expenditure_primary_education ,na.rm = TRUE)
df$infant_mortality_rate_per_1000[is.na(df$infant_mortality_rate_per_1000)] <- mean(df$infant_mortality_rate_per_1000,na.rm = TRUE)
df$per_adult_literacy_rate[is.na(df$per_adult_literacy_rate)] <- mean(df$per_adult_literacy_rate,na.rm = TRUE)
df$per_annual_population_growth[is.na(df$per_annual_population_growth)] <- median(df$per_annual_population_growth,na.rm = TRUE)
df$total_population[is.na(df$total_population)] <- median(df$total_population,na.rm = TRUE)
df$per_primary_completion[is.na(df$per_primary_completion)] <- mean(df$per_primary_completion,na.rm = TRUE)
df$per_gdp_health_expenditure[is.na(df$per_gdp_health_expenditure)] <- mean(df$per_gdp_health_expenditure,na.rm = TRUE)
df$health_expenditure_per_capita[is.na(df$health_expenditure_per_capita)] <- mean(df$health_expenditure_per_capita,na.rm = TRUE)
df$per_unemployment[is.na(df$per_unemployment)] <- median(df$per_unemployment,na.rm = TRUE)
df$per_adult_female_mortality_rate_per_1000[is.na(df$per_adult_female_mortality_rate_per_1000)] <- mean(df$per_adult_female_mortality_rate_per_1000,na.rm = TRUE)
df$per_adult_male_mortality_rate_per_1000[is.na(df$per_adult_male_mortality_rate_per_1000)] <- mean(df$per_adult_male_mortality_rate_per_1000,na.rm = TRUE)
df$per_annual_gdp_growth[is.na(df$per_annual_gdp_growth)] <- mean(df$per_annual_gdp_growth,na.rm = TRUE)
df$gdp_per_capita[is.na(df$gdp_per_capita)] <- median(df$gdp_per_capita,na.rm = TRUE)
df$gni_per_capita[is.na(df$gni_per_capita)] <- mean(df$gni_per_capita,na.rm = TRUE)
df$net_national_income[is.na(df$net_national_income )] <- mean(df$net_national_income ,na.rm = TRUE)
df$employment_to_population_ratio[is.na(df$employment_to_population_ratio)] <- mean(df$employment_to_population_ratio,na.rm = TRUE)


# checking if the imputation is done for all the columns
colSums(sapply(df, is.na))
str(df)

#df = subset(mydata, select = -c(x,z))
df_new.corr<-cor(df)
df_new.corr
corrplot.mixed(df_new.corr, lower.col = "black", number.cex = .6)

#building full model
full_model <- glm(life_expectancy_at_birth ~.,data=df)

#calculating VIF values of variables in current model
vif_values <- vif(full_model)

#displaying vif values
vif_values

#Plotting VIF values 
barplot(vif_values, main = "VIF Values", horiz = TRUE, col = "steelblue")
abline(v =5, lwd =3, lty =2)


