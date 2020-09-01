# coding: utf-8
import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
print(df)

df["len"] = df['text'].str.len()

print(df['text'].str.split().str.len())

print(df['text'].str.contains('appointment'))

print(df['text'].str.count(r'\d'))

print(df['text'].str.findall(r'\d'))

print(df['text'].str.findall(r'(\d?\d):(\d\d)'))

print(df['text'].str.replace(r'\w+day\b', '???'))

print(df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3]))

df2 = df['text'].str.extract(r'(\d?\d):(\d\d)')

print(df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))'))

print(df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))'))

