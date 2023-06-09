# try out chatgpt for sentiment analysis
# this is only meant to run once because openai only allows 60 queries a minute to chatgpt
# we're going to take the first 300 tweets to create a new csv file and then do the accuracy analysis in the main program

import openai
import time
import configparser
import pandas as pd

# load in Sentiment_annotations.csv and api_key for chatgpt
config = configparser.ConfigParser()

# people do not need to know my openai key
file_path = '/Users/carissaslone/Desktop/config.ini'
config.read(file_path)
openai.api_key = config.get('APIKEY','key') 

model_engine = "text-davinci-003" # select chatgpt model

# read in csv file with annotated values for accuracy analysis
df = pd.read_csv("test.csv")
df = df.head(300) # keep only the first 300 rows

# new column for chatgpt labels
df['chatgpt'] = None

# run through rows to call chatgpt and classify sentiment
for index, row in df.iterrows():
    entry = row['text']

    prompt = "Classify the sentiment of this sentence\n" + entry + "\n" + "as positive, negative, or neutral. Respond with one word in all lowercase."

    # call chatgpt and get sentiment label
    completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=15,
    n=1,
    stop=None,
    temperature=0.5,
    )
    
    response = completion.choices[0].text.strip()

    # write sentiment label to chatgpt column
    df['chatgpt'] = response

    # wait 1 second because openai only allows 60 queries/min
    time.sleep(1)

# create new csv file
df.to_csv('chatgpt_version.csv', index=False)