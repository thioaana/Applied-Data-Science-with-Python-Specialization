# Assignment 2 - Pandas Introduction
# Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on
# All Time Olympic Games Medals, and does some basic data cleaning.
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals,
# total # number of games, total # of medals. Use this dataset to answer the questions below.

import pandas as pd
import sys

def answer_zero():
    # What is the first country in df?
    # This function should return a Series.
    return df.iloc[0]

def answer_one():
    # Which country has won the most gold medals in summer games?
    # This function should return a single string value.*
    return df[df["Gold"] == max(df["Gold"])].index.tolist()[0]

def answer_two():
    # Which country had the biggest difference between their summer and winter gold medal counts?
    # This function should return a single string value.
    return df[df["Gold"] - df["Gold.1"] == (df["Gold"] - df["Gold.1"]).max()].index.tolist()[0]

def answer_three():
    # Which country has the biggest difference between their summer gold medal counts and
    # winter gold medal counts relative to their total gold medal count?
    # (Summer Gold − Winter Gold) / Total Gold
    # Only include countries that have won at least 1 gold in both summer and winter.
    # This function should return a single string value.
    df2 = df[(df["Gold"] > 0) & (df["Gold.1"] > 0)]
    return df2[(df2["Gold"] - df2["Gold.1"]) / df2["Gold.2"] ==
            ((df2["Gold"] - df2["Gold.1"]) / df2["Gold.2"]).max()].index.tolist()[0]

def answer_four():
    # Write a function that creates a Series called "Points" which is a weighted value # where each gold medal
    # (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point.
    # The function should return only the column (a Series object) which you created, with the country names as indices.
    # This function should return a Series named `Points` of length 146*
    df["Points"] = df["Gold.2"] * 3 + df["Silver.2"] * 2 + df["Bronze.2"] * 1
    return df["Points"]

def answer_five():
    # Which state has the most counties in it? (hint: consider the sumlevel key carefully!
    # You'll need this for future questions too...)
    # This function should return a single string value.
    df = census_df[census_df["SUMLEV"] == 50]
    df = df.groupby(["STNAME"]).count()
    first = df[df["SUMLEV"] == df["SUMLEV"].max()]
    return first.index[0]

def answer_six():
    # Only looking at the three most populous counties for each state, what are the three
    # most populous states (in order of highest population to lowest population)? Use CENSUS2010POP.
    # This function should return a list of string values.
    state3_df = census_df[census_df["SUMLEV"] == 40]
    state3_df = state3_df[["STNAME", "CENSUS2010POP"]]
    state3_df = state3_df.set_index(["STNAME"])
    state3_df["CENSUS2010POP"] = 0

    df = census_df[census_df["SUMLEV"] == 50]
    df = df[["STNAME", "CTYNAME", "CENSUS2010POP"]]
    df = df.set_index(["STNAME", "CENSUS2010POP"]).sort_index(ascending = False)
    # print(df.head(35));sys.exit()
    for st in state3_df.index:
        n = 0
        for stn in df.index:
            if st == stn[0] :
                state3_df.loc[st]["CENSUS2010POP"] += stn[1]
                n += 1
                # print(st, stn)

                if n == 3:
                    # print(st, state3_df.loc[st]["CENSUS2010POP"])
                    # sys.exit()
                    break

    state3_df.reset_index(inplace = True)
    state3_df = state3_df.set_index(["CENSUS2010POP"])
    state3_df = state3_df.sort_index(ascending = False)
    return state3_df.iloc[:3]["STNAME"].tolist()

def answer_seven():
    # Which county has had the largest absolute change in population within the period 2010-2015?
    # (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
    # e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130,
    # then its largest change in the period would be |130-80| = 50.
    # This function should return a single string value.
    columns_to_keep = [ 'CTYNAME',
                        'POPESTIMATE2010',
                        'POPESTIMATE2011',
                        'POPESTIMATE2012',
                        'POPESTIMATE2013',
                        'POPESTIMATE2014',
                        'POPESTIMATE2015']
    df = census_df[census_df["SUMLEV"] == 50]
    df = df[columns_to_keep]
    df = df.set_index(["CTYNAME"])
    df["max6"] = df.max(axis=1)
    df["min6"] = df.min(axis=1)
    df["dif"] = df["max6"] - df["min6"]
    df.reset_index(inplace = True)
    df = df.set_index(["dif"]).sort_index(ascending = False)
    return df.iloc[0,0]

def answer_eight():
    # In this datafile, the United States is broken up into four regions using the "REGION" column.
    # Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington',
    # and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
    # *This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID
    # as the census_df (sorted ascending by index).*
    df = census_df[(census_df["SUMLEV"] == 50) & (census_df["POPESTIMATE2015"] > census_df["POPESTIMATE2014"])]
    df = df[(df["REGION"] == 1) | (df["REGION"] == 2)]
    df = df[["STNAME", "CTYNAME"]]
    df = df[df["CTYNAME"].str.startswith('Washington')]
    return df.sort_index(ascending = True)

if __name__ == '__main__':
    # PART 1
    df = pd.read_csv('data/olympics.csv', index_col=0, skiprows=1)

    for col in df.columns:
        if col[:2]=='01':
            df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
        if col[:2]=='02':
            df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
        if col[:2]=='03':
            df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
        if col[:1]=='№':
            df.rename(columns={col:'#'+col[1:]}, inplace=True)

    names_ids = df.index.str.split('\s\(') # split the index by '('

    df.index = names_ids.str[0] # the [0] element is the country name (new index)
    df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

    df = df.drop('Totals')
    # print(df.head(5))

    print(answer_zero())
    print("answer 1\n-------------\n", answer_one())
    print("answer 2\n-------------\n", answer_two())
    print("answer 3\n-------------\n", answer_three())
    print("answer 4\n-------------\n", answer_four())

    # PART2
    census_df = pd.read_csv('data/census.csv')

    print("answer 5\n-------------\n", answer_five())
    print("answer 6\n-------------\n", answer_six())
    print("answer 7\n-------------\n", answer_seven())
    print("answer 8\n-------------\n", answer_eight())