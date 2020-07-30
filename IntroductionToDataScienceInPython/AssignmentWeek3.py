import pandas as pd
import numpy as np
import xlrd, sys, re

# energyDf = pd.DataFrame()

def answer_one():
    # Join th/y the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).
    # The index of this DataFrame should be the name of the country, and the columns should be
    # ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index',
    # 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012',
    # '2013', '2014', '2015'].
    energyDf = pd.read_excel("data/EnergyIndicators.xls", skiprows = 17)
    columnsToKeep = ["Country", "Energy Supply", "Energy Supply per Capita", "% Renewable"]
    energyDf.columns = ["Del0", "Del1"] + columnsToKeep
    energyDf = energyDf[columnsToKeep]
    energyDf = energyDf.iloc[:227]

    # For all countries which have missing data (e.g. data with "...") make sure this is reflected as np.NaN values.
    for ind, row in energyDf.iterrows():
        s1 = energyDf.iloc[ind, 0]
        s2 = ''.join([i for i in s1 if not i.isdigit()])
        energyDf.iloc[ind, 0] = re.sub(r"[\(\[].*?[\)\]]", "", s2)
        if "..." in str(row["Energy Supply"]):
            energyDf.iloc[ind, 1] = np.nan
        if "..." in str(row["Energy Supply per Capita"]):
            energyDf.iloc[ind, 2] = np.nan
        if "..." in str(row["% Renewable"]):
            energyDf.iloc[ind, 3] = np.nan

    energyDf.replace({"Republic of Korea": "South Korea",
                      "United States of America" : "United States",
                      "United Kingdom of Great Britain and Northern Ireland" : "United Kingdom",
                      "China, Hong Kong Special Administrative Region" : "Hong Kong"},
                      inplace = True)

    # Convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule).
    energyDf["Energy Supply"] = energyDf["Energy Supply"] * 1000000

    GDPdf = pd.read_csv("data/world_bank.csv", skiprows = 4)
    GDPdf.replace({"Korea, Rep." : "South Korea",
                   "Iran, Islamic Rep." : "Iran",
                   "Hong Kong SAR, China" : "Hong Kong"},
                   inplace = True)

    # Use only the last 10 years (2006-2015) of GDP data
    GDPcolumnsToKeep = ["Country Name", "Country Code", "2006", "2007", "2008", "2009", '2010', '2011',
    '2012', '2013', '2014', '2015']
    GDPdf = GDPdf[GDPcolumnsToKeep]

    ScimEnDf = pd.read_excel("data/scimagojr-3.xlsx")
    ScimEnDf2 = ScimEnDf.iloc[:15] # Use only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15).

    energyDf2 = energyDf.set_index(["Country"])
    GDPdf2 = GDPdf.set_index(["Country Name"])
    ScimEnDf2 = ScimEnDf2.set_index(["Country"])

    df1 = pd.merge(energyDf2, GDPdf2, how = "outer", left_index = True, right_index = True)
    dfColumnsToKeep = ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document',
                        'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009',
                        '2010', '2011', '2012', '2013', '2014', '2015']
    df2 = pd.merge(df1, ScimEnDf2, how = "inner", left_index = True, right_index = True)
    df2 = df2[dfColumnsToKeep]
    return df2

def answer_two():
    # The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets,
    # but before you reduced this to the top 15 items, how many entries did you lose?
    # This function should return a single number. The request is phrased completely wrong.
    energyDf = pd.read_excel("data/EnergyIndicators.xls", skiprows = 17)
    columnsToKeep = ["Country", "Energy Supply", "Energy Supply per Capita", "% Renewable"]
    energyDf.columns = ["Del0", "Del1"] + columnsToKeep
    energyDf = energyDf[columnsToKeep]
    energyDf = energyDf.iloc[:227]

    # For all countries which have missing data (e.g. data with "...") make sure this is reflected as np.NaN values.
    for ind, row in energyDf.iterrows():
        s1 = energyDf.iloc[ind, 0]
        s2 = ''.join([i for i in s1 if not i.isdigit()])
        energyDf.iloc[ind, 0] = re.sub(r"[\(\[].*?[\)\]]", "", s2)
        if "..." in str(row["Energy Supply"]):
            energyDf.iloc[ind, 1] = np.nan
        if "..." in str(row["Energy Supply per Capita"]):
            energyDf.iloc[ind, 2] = np.nan
        if "..." in str(row["% Renewable"]):
            energyDf.iloc[ind, 3] = np.nan
        energyDf.iloc[ind,0] = str(energyDf.iloc[ind,0]).rstrip()

    energyDf.replace({"Republic of Korea": "South Korea",
                      "United States of America" : "United States",
                      "United Kingdom of Great Britain and Northern Ireland" : "United Kingdom",
                      "China, Hong Kong Special Administrative Region" : "Hong Kong"},
                      inplace = True)

    # Convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule).
    energyDf["Energy Supply"] = energyDf["Energy Supply"] * 1000000

    GDPdf = pd.read_csv("data/world_bank.csv", skiprows = 4)
    GDPdf.replace({"Korea, Rep." : "South Korea",
                   "Iran, Islamic Rep." : "Iran",
                   "Hong Kong SAR, China" : "Hong Kong"},
                   inplace = True)

    # Use only the last 10 years (2006-2015) of GDP data
    GDPcolumnsToKeep = ["Country Name", "Country Code", "2006", "2007", "2008", "2009", '2010', '2011',
    '2012', '2013', '2014', '2015']
    GDPdf = GDPdf[GDPcolumnsToKeep]

    ScimEnDf = pd.read_excel("data/scimagojr-3.xlsx")

    energyDf2 = energyDf.set_index(["Country"])
    GDPdf2 = GDPdf.set_index(["Country Name"])
    ScimEnDf2 = ScimEnDf.set_index(["Country"])

    dfo1 = pd.merge(energyDf2, GDPdf2, how = "outer", left_index = True, right_index = True)
    dfo2 = pd.merge(dfo1, ScimEnDf2, how = "outer", left_index = True, right_index = True)
    dfi1 = pd.merge(energyDf2, GDPdf2, how = "inner", left_index = True, right_index = True)
    dfi2 = pd.merge(dfi1, ScimEnDf2, how = "inner", left_index = True, right_index = True)

    dfColumnsToKeep = ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document',
                        'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009',
                        '2010', '2011', '2012', '2013', '2014', '2015']

    dfo2 = dfo2[dfColumnsToKeep]
    dfi2 = dfi2[dfColumnsToKeep]
    return dfo2.shape[0] - dfi2.shape[0]

def answer_three():
    # What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
    # This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order
    def aver(row):
        data = row[["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"]]
        return pd.Series(np.mean(data))
    Top15 = answer_one()
    Top15["avgGDP"] = Top15.apply(aver, axis=1)
    Top15 = Top15.sort_values(by=["avgGDP"], ascending = False)
    return Top15["avgGDP"]

def answer_four():
    # By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
    # This function should return a single number.
    Top15 = answer_one()
    countryName6 = answer_three().reset_index().iloc[5, 0]
    recordAvg6 = Top15.loc[countryName6]
    return recordAvg6["2015"]-recordAvg6["2006"]

def answer_five():
    # What is the mean Energy Supply per Capita?
    # This function should return a single number.
    Top15 = answer_one()
    return Top15["Energy Supply per Capita"].mean().tolist()

def answer_six():
    # What country has the maximum % Renewable and what is the percentage?
    # This function should return a tuple with the name of the country and the percentage.
    Top15 = answer_one()
    maxRen = Top15.reset_index().sort_values(by = "% Renewable", ascending = False).iloc[0]
    return (maxRen[0], maxRen[10])

def answer_seven():
    # Create a new column that is the ratio of Self-Citations to Total Citations. What is the maximum value
    # for this new column, and what country has the highest ratio?
    # This function should return a tuple with the name of the country and the ratio.
    Top15 = answer_one()
    Top15["% Citations"] = Top15["Self-citations"] / Top15["Citations"]
    maxCit = Top15.reset_index().sort_values(by = "% Citations", ascending = False).iloc[0]
    return (maxCit[0], maxCit[21])

def answer_eight():
    # Create a column that estimates the population using Energy Supply and Energy Supply per capita.
    # What is the third most populous country according to this estimate?
    # This function should return a single string value.
    Top15 = answer_one()
    Top15["myPopul"] = Top15["Energy Supply"] / Top15["Energy/Capita"]
    # print(Top15.sort_values(by = "myPopul", ascending = False)["myPopul"])
    maxPopul = Top15.reset_index().sort_values(by = "myPopul", ascending = False).iloc[2]
    return maxPopul[0]

def answer_nine():
    # Create a column that estimates the number of citable documents per person.
    # What is the correlation  between the number of citable documents per capita and the energy supply per capita?
    # Use the .corr() method, (Pearson's correlation).
    # This function should return a single number.
    Top15 = answer_one()
    Top15["Citable/Capita"] = (Top15["Energy Supply"] / Top15["Energy/Capita"]) / Top15["Citable documents"]
    Top15["Citable/Capita"] = Top15["Citable/Capita"].astype(float)
    Top15["Energy/Capita"] = Top15["Energy/Capita"].astype(float)
    return Top15["Citable/Capita"].corr(Top15["Energy/Capita"], method = 'pearson')

def answer_ten():
    # Create a new column with a 1 if the country's % Renewable value is at or above the median
    # for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
    # This function should return a series named `HighRenew` whose index is the country name sorted
    # in ascending order of rank.*
    Top15 = answer_one()
    Top15["HighRenew"] = 1
    med = Top15["% Renewable"].median()
    Top15["HighRenew"][Top15["% Renewable"] < med] = 0
    # Top15["HighRenew"][Top15["% Renewable"] == np.nan] = np.nan
    Top15.sort_values(by = ["Rank"], ascending = True, inplace = True)
    # print(Top15[["% Renewable", "HighRenew"]])
    # print("\n\n",Top15[["% Renewable", "HighRenew"]].sort_values(by = ["% Renewable"], ascending=True), med)
    return Top15["HighRenew"]

def answer_eleven():
    # Use the following dictionary to group the Countries by Continent, then create a dateframe that displays
    # the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for
    # the estimated population of each country.
    ContinentDict  = {'China':'Asia',
                      'United States':'North America',
                      'Japan':'Asia',
                      'United Kingdom':'Europe',
                      'Russian Federation':'Europe',
                      'Canada':'North America',
                      'Germany':'Europe',
                      'India':'Asia',
                      'France':'Europe',
                      'South Korea':'Asia',
                      'Italy':'Europe',
                      'Spain':'Europe',
                      'Iran':'Asia',
                      'Australia':'Australia',
                      'Brazil':'South America'}
    # *This function should return a DataFrame with index named Continent
    # `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`
    Top15 = answer_one().dropna()
    Top15["Continent"] = Top15.index
    Top15["Continent"].replace(ContinentDict, inplace = True)
    Top15["myPopul"] = Top15["Energy Supply"] / Top15["Energy/Capita"]
    Cont15 = Top15[["Continent", "myPopul"]]
    Cont15["myPopul"] = Cont15["myPopul"].astype(int)

    return Cont15.set_index("Continent").groupby(level=0)["myPopul"].agg({"size" : np.size, "sum" : np.sum, "mean" : np.mean, "std" : np.std})

def answer_twelve():
    # Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins.
    # How many countries are in each of these groups?
    # *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`.
    # Do not include groups with no countries.*
    ContinentDict  = {'China':'Asia', 'United States':'North America', 'Japan':'Asia', 'United Kingdom':'Europe',
                      'Russian Federation':'Europe', 'Canada':'North America', 'Germany':'Europe',
                      'India':'Asia', 'France':'Europe', 'South Korea':'Asia', 'Italy':'Europe',
                      'Spain':'Europe', 'Iran':'Asia', 'Australia':'Australia', 'Brazil':'South America'}
    Top15 = answer_one().dropna()
    Top15["Continent"] = Top15.index
    Top15["Continent"].replace(ContinentDict, inplace = True)
    Cont = pd.cut(Top15["% Renewable"], 5)

    Cont15 = Top15[["Continent", "% Renewable"]]
    Cont15["Cat Renewable"] = pd.cut(Cont15["% Renewable"], 5)
    print(Cont15);sys.exit()

    print(Cont15.groupby(["Continent", "Cat Renewable"], as_index=False));sys.exit()
    Cont15 = Cont15.set_index("Continent", "Cat Renewable")
    # Cont15 = Cont15.set_index([])
    # print(Cont15)
    print(Cont15.set_index("Continent", "Cat Renewable").groupby(level=0))
    # print(Cont15.groupby(["Continent", "Cat Renewable"]))
    # print(Cont15)
    # print(Top15.iloc[0])
    return "ANSWER"

def answer_thirteen(Top15):
    # Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
    # e.g. 317615384.61538464 -> 317,615,384.61538464
    # *This function should return a Series `PopEst` whose index is the country name and whose values are
    # the population estimate string.*
    return "ANSWER"

if __name__ == '__main__':
    # Load the energy data from the file Energy Indicators.xls, which is a list of indicators of energy supply and
    # renewable electricity production from the United Nations for the year 2013, and should be put into a DataFrame
    # with the variable name of energy.
    # Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude
    # the footer and header information from the datafile. The first two columns are unneccessary, so you should
    # get rid of them, and you should change the column labels so that the columns are:
    # ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    #
    # energyDf = pd.read_excel("data/EnergyIndicators.xls", skiprows = 17)
    # columnsToKeep = ["Country", "Energy Supply", "Energy/Capita", "% Renewable"]
    # energyDf.columns = ["Del0", "Del1"] + columnsToKeep
    # energyDf = energyDf[columnsToKeep]
    # energyDf = energyDf.iloc[:227]
    #
    # # Convert Energy Supply to gigajoules (there are 1,000,000 gigajoules in a petajoule).
    # energyDf["Energy Supply"] = energyDf["Energy Supply"] * 1000000
    #
    # # There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these,
    # # e.g.
    # # 'Bolivia (Plurinational State of)' should be 'Bolivia',
    # # 'Switzerland17' should be 'Switzerland'.
    # for ind, row in energyDf.iterrows():
    #     s1 = energyDf.iloc[ind, 0]
    #     s2 = ''.join([i for i in s1 if not i.isdigit()])
    #     energyDf.iloc[ind, 0] = re.sub(r"[\(\[].*?[\)\]]", "", s2)
    #
    # # For all countries which have missing data (e.g. data with "...") make sure this is reflected as np.NaN values.
    # energyDf = energyDf.replace(r'^[.]', np.nan)
    #
    # # Rename the following list of countries (for use in later questions):
    # # "Republic of Korea": "South Korea",
    # # "United States of America": "United States",
    # # "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    # # "China, Hong Kong Special Administrative Region": "Hong Kong"
    # energyDf.replace({"Republic of Korea": "South Korea",
    #                   "United States of America" : "United States",
    #                   "United Kingdom of Great Britain and Northern Ireland" : "United Kingdom"}, inplace = True)
    # energyDf.replace("China, Hong Kong Special Administrative Region", "Hong Kong", inplace = True)
    # # print("England \n",energyDf[energyDf["Country"] == "United Kingdom"]);sys.exit()
    #
    # # print("energyDf\n----------\n",energyDf.head(5))
    #
    #
    # # Next, load the GDP data from the file world_bank.csv, which is a csv containing countries' GDP from 1960 to 2015
    # # from World Bank. Call this DataFrame GDP.Make sure to skip the header, and rename the following list of countries:
    # # "Korea, Rep.": "South Korea",
    # # "Iran, Islamic Rep.": "Iran",
    # # "Hong Kong SAR, China": "Hong Kong"
    # GDPdf = pd.read_csv("data/world_bank.csv", skiprows = 4)
    # GDPdf = GDPdf.replace("Korea, Rep.", "South Korea")
    # GDPdf = GDPdf.replace("Iran, Islamic Rep.", "Iran")
    # GDPdf = GDPdf.replace("Hong Kong SAR, China", "Hong Kong")
    # # Use only the last 10 years (2006-2015) of GDP data
    # GDPcolumnsToKeep = ["Country Name", "Country Code", "2006", "2007", "2008", "2009", '2010', '2011',
    # '2012', '2013', '2014', '2015']
    # GDPdf = GDPdf[GDPcolumnsToKeep]
    # # print("GDPdf\n-----------\n",GDPdf.head(5))
    #
    # # Finally, load the Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology from the
    # # file scimagojr-3.xlsx, which ranks countries based on their journal contributions in the aforementioned area.
    # # Call this DataFrame ScimEn.
    # ScimEnDf = pd.read_excel("data/scimagojr-3.xlsx")
    # print("ScimEnDf\n--------------\n",ScimEnDf)

    Top15 = answer_one()
    print("\nAnswer 1 :\n---------\n",answer_one())
    print("\nAnswer 2 :\n---------\n",answer_two())
    print("\nAnswer 3 :\n---------\n",answer_three())
    print("\nAnswer 4 :", answer_four(), "\n-------------------------")
    print("\nAnswer 5 :", answer_five(), "\n------------------------")
    # print("\nAnswer 6 :", answer_six(), "\n------------------------")
    # print("\nAnswer 7 :", answer_seven(), "\n------------------------")
    # print("\nAnswer 8 :", answer_eight(), "\n------------------------")
    # print("\nAnswer 9 :", answer_nine(), "\n------------------------")
    # print("\nAnswer 10 :", "\n------------------------\n", answer_ten())
    # print("\nAnswer 11 :", "\n------------------------\n", answer_eleven())
    # print("\nAnswer 12 :", "\n------------------------\n", answer_twelve())