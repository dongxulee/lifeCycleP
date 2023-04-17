import pandas as pd
pd.options.display.float_format = "{:.2f}".format
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

# All variables we care about
FcolumnNames1999_2007 = ['releaseNum', 'familyID', 'composition', 'headCount', 'ageHead', 'maritalStatus', 'own', 
                         'employmentStatus', 'liquidWealth', 'race', 'industry','incomeHead', "incomeWife", 
               'foodCost', 'houseCost', 'transCost', 'educationCost', 'childCost', 'healthCost', 'education', 
               'participation', 'investmentAmount', 'annuityIRA', 'wealthWithoutHomeEquity', "wealthWithHomeEquity"]

FcolumnNames2009_2017 = ['releaseNum', 'familyID', 'composition', 'headCount', 'ageHead', 'maritalStatus', 'own', 
                         'employmentStatus', 'liquidWealth', 'race', 'industry' ,'incomeHead', 'incomeWife', 
               'participation', 'investmentAmount', 'annuityIRA', 'wealthWithoutHomeEquity', 'wealthWithHomeEquity',
               'foodCost', 'houseCost', 'transCost', 'educationCost', 'childCost', 'healthCost', 'education']

FcolumnNames2019 = ['releaseNum', 'familyID', 'composition', 'headCount', 'ageHead', 'maritalStatus', 'own', 
                         'employmentStatus', 'liquidWealth_bank', 'liquidWealth_bond', 'race', 'industry' ,'incomeHead', 'incomeWife', 
               'participation', 'investmentAmount', 'annuityIRA', 'wealthWithoutHomeEquity', 'wealthWithHomeEquity',
               'foodCost', 'houseCost', 'transCost', 'educationCost', 'childCost', 'healthCost', 'education']

# The timeline we care about
years = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]

def Fcompile_data_with_features(features, years = years):
    df = pd.DataFrame()
    # Loading the data through years
    for year in years:
        df_sub = pd.read_csv(str(year) + ".csv")
        if year >= 1999 and year <= 2007:
            df_sub.columns = FcolumnNames1999_2007
        elif year >= 2009 and year <= 2017:
            df_sub.columns = FcolumnNames2009_2017
        else:
            # In the year 2019, the liquid wealth equals to liquidWealth in cash and liquid bond
            df_sub.columns = FcolumnNames2019
            df_sub["liquidWealth"] = df_sub['liquidWealth_bank'] + df_sub['liquidWealth_bond']
        df_sub['year'] = year
        df = pd.concat([df, df_sub[['familyID','year'] + features]])
    df = df.reset_index(drop = True)
    return df

# The function is used to drop the values we do not like in the dataFrame, 
# the input "features" and "values" are both list
def drop_values(features, values, df): 
    for feature in features:
        for value in values:
            df = df[df[feature] != value]
    df = df.reset_index(drop = True)
    return df

# prepare the combined dataset and set up dummy variables for qualitative data
df = Fcompile_data_with_features(['composition', 'headCount', 'ageHead', 'maritalStatus', 'own',
                                  'employmentStatus', 'liquidWealth', 'race', 'industry','incomeHead', 'incomeWife', 
                                  'foodCost', 'houseCost', 'transCost', 'educationCost', 'childCost', 'healthCost', 'education', 
                                  'participation', 'investmentAmount', 'annuityIRA', 'wealthWithoutHomeEquity', 'wealthWithHomeEquity'], years)

# data clean, drop NA/DK values
df = drop_values(["ageHead"],[999], df)
df = drop_values(["maritalStatus"],[8,9], df)
df = drop_values(["own"],[8,9], df)
df = drop_values(["employmentStatus"],[0,22,8,98, 99], df)
df = drop_values(["liquidWealth"],[999999998,999999999,-400], df)
df = drop_values(["race"],[0,8,9], df)
df = drop_values(["industry"],[999,9999,0], df)
df = drop_values(["education"],[99,0], df)
# calculate the aggregate variables 
df["totalExpense"] = df[['foodCost', 'houseCost', 'transCost', 
                                      'educationCost', 'childCost', 'healthCost']].sum(axis = 1)
df["laborIncome"] = df["incomeHead"] + df["incomeWife"]
df["costPerPerson"] = df["totalExpense"]/df["headCount"]
df["HomeEquity"] = df["wealthWithHomeEquity"] - df["wealthWithoutHomeEquity"]

maritalStatus = ["Married", "neverMarried", "Widowed", "Divorced", "Separated"]
employmentStatus = ["Working", "temporalLeave", "unemployed", "retired", "disabled", "keepHouse", "student", "other"]
race = ["White", "Black","AmericanIndian","Asian","Latino","otherBW","otherRace"]
# Education
# < 8th grade: middle school
# >= 8 and < 12: high scho0l
# >=12 and < 15: college
# >= 15 post graduate
education = ["middleSchool", "highSchool", "college", "postGraduate"]
# Industry
# < 400 manufacturing
# >= 400 and < 500 publicUtility
# >= 500 and < 680 retail 
# >= 680 and < 720 finance
# >= 720 and < 900 service
# >= 900 otherIndustry
industry = ["finance", "noneFinance"]
ownership = ["owner", "renter"]

data = []
for i in tqdm(range(len(df))):
    dataCollect = []
    # marital status
    dataCollect.append(maritalStatus[int(df.iloc[i]["maritalStatus"]-1)])
    # employment
    dataCollect.append(employmentStatus[int(df.iloc[i]["employmentStatus"]-1)])
    # race
    dataCollect.append(race[int(df.iloc[i]["race"] - 1)])
    # Education variable 
    if df.iloc[i]["education"] < 8:
        dataCollect.append(education[0])
    elif df.iloc[i]["education"] >= 8 and df.iloc[i]["education"] < 12:
        dataCollect.append(education[1])
    elif df.iloc[i]["education"] >= 12 and df.iloc[i]["education"] < 15:
        dataCollect.append(education[2])
    else:
        dataCollect.append(education[3])
    # industry variable 
    if df.iloc[i]["year"] in [1999, 2001]:
        if df.iloc[i]["industry"] >= 707 and df.iloc[i]["industry"] <= 718:
            dataCollect.append(industry[0])
        else:
            dataCollect.append(industry[1])
    elif df.iloc[i]["year"] in [2003,2005,2007,2009,2011,2013,2015,2015]:
        if df.iloc[i]["industry"] >= 687 and df.iloc[i]["industry"] <= 699:
            dataCollect.append(industry[0])
        else:
            dataCollect.append(industry[1])        
    else:
        if df.iloc[i]["industry"] >= 6870 and df.iloc[i]["industry"] <= 6990:
            dataCollect.append(industry[0])
        else:
            dataCollect.append(industry[1])
    # ownership status 
    if df.iloc[i]["own"] == 1:
        dataCollect.append(ownership[0])
    else:
        dataCollect.append(ownership[1])
    data.append(dataCollect)
# Categorical dataFrame
df_cat = pd.DataFrame(data, columns = ["maritalStatus", "employmentStatus", "race", "education", "industry", "ownership"])

Fdf = pd.concat([df[["familyID", "year",'composition', 'headCount', 'ageHead', 'liquidWealth', 'laborIncome', 
                     "costPerPerson","totalExpense", 'participation', 'investmentAmount', 'annuityIRA', 
                                 'wealthWithoutHomeEquity', "wealthWithHomeEquity", "HomeEquity"]], 
                          df_cat[["maritalStatus", "employmentStatus", "education","race", "industry", "ownership"]]], axis=1)

# Adjust for inflation, all values are in thousand dollor
years = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]
values_at2020 = np.array([1.55, 1.46, 1.40, 1.32, 1.24, 1.20, 1.15, 1.11, 1.09, 1.05, 1.01])
values_at2005 = values_at2020/1.32
quantVariables = ['annuityIRA', 'investmentAmount', 'liquidWealth', 'laborIncome', 'costPerPerson',
                 'totalExpense', 'wealthWithoutHomeEquity', 'wealthWithHomeEquity', "HomeEquity"]
for i in tqdm(range(len(Fdf))):
    for variable in quantVariables:
        Fdf.loc[i, variable] = round(Fdf.loc[i, variable] * values_at2005[years.index(Fdf.loc[i,"year"])] / 1000, 2)

# drop the extreme outliers 
for var in quantVariables:
    Fdf = Fdf[Fdf[var] < Fdf[var].quantile(0.999)]
Fdf = Fdf[(Fdf["ageHead"] >= 20) & (Fdf["ageHead"] <= 80)]

# group the population into 4 types of agents ]
lowSkill = ["middleSchool", "highSchool"]
highSkill = ["college", "postGraduate"]
highFinance = Fdf[(Fdf["education"].isin(highSkill)) & (Fdf["industry"] == "finance")]
lowFinance = Fdf[(Fdf["education"].isin(lowSkill)) & (Fdf["industry"] == "finance")]
highNoneFinance = Fdf[(Fdf["education"].isin(highSkill)) & (Fdf["industry"] == "noneFinance")]
lowNoneFinance = Fdf[(Fdf["education"].isin(lowSkill)) & (Fdf["industry"] == "noneFinance")]

Fdf["skillLevel"] = "High"
Fdf.loc[Fdf["education"].isin(lowSkill), "skillLevel"] = "Low"
Fdf["financeExperience"] = "No"
Fdf.loc[Fdf["industry"] == "finance", "financeExperience"] = "Yes"
Fdf["ageGroup"] = "20"
Fdf["decadeGroup"] = "90's"
for i in range(2,10, 2):
    Fdf.loc[Fdf["ageHead"] > i*10, "ageGroup"] = str(i*10)
for year in range(1990,2020,10):
    Fdf.loc[Fdf["year"] > year, "decadeGroup"] = str(year) + "s"
    
Fdf.loc[(Fdf["employmentStatus"] != "Working")&(Fdf["employmentStatus"] != "retired"), "employmentStatus"] = "unemployed"
Fdf.loc[Fdf["employmentStatus"]=="Working", "employmentStatus"] = "employed"

Fdf.loc[Fdf["ageGroup"]== "20", "ageGroup"] = "20-40"
Fdf.loc[Fdf["ageGroup"]== "40", "ageGroup"] = "40-60"
Fdf.loc[Fdf["ageGroup"]== "60", "ageGroup"] = "60-80"

Fdf["stockInvestmentRatio"] = Fdf.investmentAmount/Fdf.wealthWithoutHomeEquity
Fdf.loc[-((Fdf["stockInvestmentRatio"] >= 0)&(Fdf["stockInvestmentRatio"] <= 1)), "stockInvestmentRatio"] = 0

print(Fdf.head())
print(Fdf.tail())

Fdf.to_csv("familyData.csv")