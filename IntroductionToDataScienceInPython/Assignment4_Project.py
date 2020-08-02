import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Use this dictionary to map state names to two letter acronyms
states = {"OH": "Ohio", "KY": "Kentucky", "AS": "American Samoa", "NV": "Nevada", "WY": "Wyoming",
          "NA": "National", "AL": "Alabama", "MD": "Maryland", "AK": "Alaska", "UT": "Utah",
          "OR": "Oregon", "MT": "Montana", "IL": "Illinois", "TN": "Tennessee", "DC": "District of Columbia",
          "VT": "Vermont", "ID": "Idaho", "AR": "Arkansas", "ME": "Maine",
          "WA": "Washington", "HI": "Hawaii", "WI": "Wisconsin", "MI": "Michigan", "IN": "Indiana", "NJ": "New Jersey",
          "AZ": "Arizona", "GU": "Guam", "MS": "Mississippi", "PR": "Puerto Rico", "NC": "North Carolina", "TX": "Texas",
          "SD": "South Dakota", "MP": "Northern Mariana Islands", "IA": "Iowa", "MO": "Missouri", "CT": "Connecticut",
          "WV": "West Virginia", "SC": "South Carolina", "LA": "Louisiana", "KS": "Kansas", "NY": "New York",
          "NE": "Nebraska", "OK": "Oklahoma", "FL": "Florida", "CA": "California", "CO": "Colorado", "PA": "Pennsylvania",
          "DE": "Delaware", "NM": "New Mexico", "RI": "Rhode Island", "MN": "Minnesota", "VI": "Virgin Islands",
          "NH": "New Hampshire", "MA": "Massachusetts", "GA": "Georgia", "ND": "North Dakota", "VA": "Virginia"}

def get_list_of_university_towns():
    """Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character "\n". """
    myList = []
    with open("data/university_towns.txt", "r") as fp:
        Lines=fp.readlines()
    for line in Lines:
        if "[edit]" in line :
            state = line[:line.find("[edit]")]
        else :
            # if "(" not in line : print(line)
            town = line[:line.find(" (")].strip()
            myList.append([state, town])
    return pd.DataFrame(myList, columns=["State", "RegionName"])

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    GDP = pd.read_excel("data/gdplev.xls", skiprows=219, usecols=[4, 6], names=["Quarter", "GDP"])
    for i in range(1, len(GDP["GDP"] - 2)) :
        if GDP["GDP"][i] < GDP["GDP"][i - 1] and GDP["GDP"][i + 1] < GDP["GDP"][i]:
            break
    return GDP["Quarter"][i]

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3
    '''
    GDP = pd.read_excel("data/gdplev.xls", skiprows=219, usecols=[4, 6], names=["Quarter", "GDP"])
    start = get_recession_start()
    indStart = GDP[GDP["Quarter"] == start].index.tolist()[0]
    for i in range(indStart, len(GDP["GDP"] - 2)):
        if GDP["GDP"][i] > GDP["GDP"][i - 1] and GDP["GDP"][i + 1] > GDP["GDP"][i]:
            break
    return GDP["Quarter"][i + 1]

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3
    '''
    GDP = pd.read_excel("data/gdplev.xls", skiprows=219, usecols=[4, 6], names=["Quarter", "GDP"])
    start = get_recession_start()
    indStart = GDP[GDP["Quarter"] == start].index.tolist()[0]
    end = get_recession_end()
    indEnd = GDP[GDP["Quarter"] == end].index.tolist()[0]
    indMinGdp = indEnd
    minGdp = GDP["GDP"][indMinGdp]
    for i in range(indStart, indEnd):
        if GDP["GDP"][i] < minGdp :
            indMinGdp = i
            minGdp = GDP["GDP"][i]
    print("({}, {})".format(GDP["Quarter"][indStart], GDP["Quarter"][indEnd]))
    return GDP["Quarter"][indMinGdp]

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.'''

    HousingData = pd.read_csv("data/City_Zhvi_AllHomes.csv", usecols=[1, 2, *range(51,251)])

    # Change the two-letters State (ie NY) with the long name (New York)
    for i in range(len(HousingData)):
        HousingData.loc[i, "State"] = states[HousingData.loc[i, "State"]]

    # Adding nwe columns (ie "2005q3") in the data frame with zero values
    newColumns = []
    for year in range(2000, 2017):
        for qt in ["q1", "q2", "q3", "q4"] :
            newColumns.append(str(year)+qt)
    newColumns=newColumns[:-1]
    for c in newColumns :
        HousingData[c] = 0.0

    # Mapping new and old columns
    mapNewVsOldCols = {}
    for c in newColumns:
        year = c[:4]
        if c[-1] == "1" : q = ["-01", "-02", "-03"]
        if c[-1] == "2": q = ["-04", "-05", "-06"]
        if c[-1] == "3": q = ["-07", "-08", "-09"]
        if c[-1] == "4": q = ["-10", "-11", "-12"]
        mapNewVsOldCols[c] = [year+q[0], year+q[1], year+q[2]]
    del mapNewVsOldCols["2016q3"]

    # Filing values in new Columns
    for k, v in mapNewVsOldCols.items() :
        HousingData[k] = (HousingData[v[0]] + HousingData[v[1]] + HousingData[v[2]]) / 3
    HousingData["2016q3"] = (HousingData["2016-07"] + HousingData["2016-08"]) / 2

    # Remove old colmns (12 months per year) and change indexing
    HousingData.drop(HousingData.iloc[:, 2:202], inplace=True, axis=1)
    HousingData.set_index(["State", "RegionName"], inplace=True)
    # print(HousingData.iloc[0:3])
    return HousingData

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    univerTowns = get_list_of_university_towns()
    univerTowns["isUnTown"] = True
    univerTowns.set_index(["State", "RegionName"], inplace=True)
    HousingData = convert_housing_data_to_quarters()
    HousingData = pd.merge(HousingData, univerTowns, how="left", left_index=True, right_index=True)
    HousingData["isUnTown"].fillna(False, inplace=True)

    recessionStart = get_recession_start()
    beforeRecessionStart = HousingData.columns.tolist()[HousingData.columns.tolist().index(recessionStart) - 1]
    recessionBottom = get_recession_bottom()
    HousingData["ratio"] = HousingData[beforeRecessionStart] / HousingData[recessionBottom]
    HousingData = HousingData[[beforeRecessionStart, recessionBottom, "ratio", "isUnTown"]]

    UT    = HousingData[HousingData["isUnTown"] == True]
    nonUT = HousingData[HousingData["isUnTown"] == False]
    (sts, pValue) = ttest_ind(UT["ratio"], nonUT["ratio"], nan_policy="omit")
    if pValue < 0.01 : difference = True
    else : difference = False
    if UT["ratio"].mean() < nonUT["ratio"].mean() : better = "university town"
    else : better = "non-university town"
    # univerTowns.to_csv("Universities.xls")
    # HousingData.to_csv("Houses.xls")
    return (difference, pValue, better)

if __name__ == "__main__" :
    # a = get_list_of_university_towns()
    # print(len(a["State"]))
    # print(a)
    # print(a.groupby("State").size())
    # print(get_recession_start())
    # print(get_recession_end())
    # print(get_recession_bottom())
    # print(convert_housing_data_to_quarters())
    print(run_ttest())
