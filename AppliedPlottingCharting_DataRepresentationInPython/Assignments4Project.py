import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EUCountries = {"Countries" :["Austria", "Belgium", "Bulgaria", "Czech Republic", "France", "Germany", "Denmark", "Greece",
               "Estonia", "Ireland", "Spain", "Italy", "Netherlands", "Croatia", "Cyprus", "Latvia", "Lithuania", "Luxembourg",
               "Malta", "Hungary", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Sweden", "Finland",
               "United Kingdom", "Iceland", "Norway", "Liechtenstein", "Switzerland"]}

# Read and prepare Population Data Frame
Population = pd.read_csv("data/PopulationPerCountry.csv")
Population = Population[["Country Name", "Country Code", "2019"]]
Population.rename(columns = {"Country Name":"Country", "Country Code" : "Code", "2019": "Pop2019"}, inplace = True)
Population.replace({"Slovak Republic" : "Slovakia", "United States" : "USA"}, inplace=True)

# Read and prepare Covid Data Frame
Covid = pd.read_csv("data/WHO-COVID-19-global-data.csv")
Covid.rename(columns={"Date_reported" : "Date", " Country_code" : "Code", " Country": "Country",
                      " WHO_region" : "Region", " New_cases": "NewCases", " Cumulative_cases": "CumulativeCases",
                      " New_deaths" : "NewDeaths", " Cumulative_deaths" :"CumulativeDeaths"}, inplace=True)
Covid.replace({"Czechia" : "Czech Republic", "The United Kingdom": "United Kingdom", "United States of America" : "USA"}, inplace=True)

# Checking that country names are the same
# EUdf = pd.DataFrame(EUCountries)
# df1 = pd.merge(EUdf, Population, how='inner', left_on='Countries', right_on='Country')
# df1.to_csv("data/temporary.csv")
# df2 = pd.merge(EUdf, Covid, how='inner', left_on='Countries', right_on='Country')
# df2.to_csv("data/temporary.csv")

# create main DF with all the countries and the Death Rates
df = pd.merge(Covid, Population, how='inner', left_on='Country', right_on='Country')
df["DeathRate"] = df.CumulativeDeaths * 1000000 / df.Pop2019

# Death Rate for Europe
EUTotalPop = Population["Pop2019"][Population.Country.isin(EUCountries["Countries"])].sum()
dfEU = df[df.Country.isin(EUCountries["Countries"])]
dfEU = dfEU.groupby("Date").sum()
dfEU["Europe"] = dfEU["CumulativeDeaths"] * 1000000 / EUTotalPop
dfEU.reset_index(inplace = True)
dfEU = dfEU[["Date", "Europe"]]

# Death Rate for Greece and USA
dfGreece = df[["Date", "DeathRate"]][df.Country == "Greece"]
dfUSA = df[df.Country == "USA"]
dfUSA = df[["Date", "DeathRate"]][df.Country == "USA"]

# Merge fra me to a final DF with index=Date and Columns=Europe, USA, Greece
dfRes = pd.merge(dfEU, dfGreece, how='inner', left_on='Date', right_on='Date')
dfRes.rename(columns = {"DeathRate":"Greece"}, inplace = True)
dfRes = pd.merge(dfRes, dfUSA, how='inner', left_on='Date', right_on='Date')
dfRes.rename(columns = {"DeathRate":"USA"}, inplace = True)
dfRes['Date'] = pd.to_datetime(dfRes['Date'])
dfRes.set_index("Date", inplace=True)

plt.style.use("seaborn-colorblind")
dfRes.plot()
plt.gca().set_title("COVID-19 - Cumulative Deaths growth\nComparison between USA, Europe and Greece")
plt.gca().set_ylabel("Cumulative Deaths per mil of population" )
plt.show()