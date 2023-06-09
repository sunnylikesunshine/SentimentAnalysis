# the specific data file has entries tagged "Irrelevant". this removes them
# also creates a much smaller data set to use for chatgpt which only allows one query a second

import pandas as pd

df = pd.read_csv('twitter_training.csv')

filtered_df = df[df['SENTIMENT'] != "Irrelevant"]

filtered_df.to_csv('filtered_twitter.csv', index=False)