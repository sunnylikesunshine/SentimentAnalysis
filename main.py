# program to classify tweets as positive/negative/neutral

import numpy as np
import pandas as pd

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from PreprocessText import PreprocessText
from SentimentCutOff import SentCutOff


# read in csv file with annotated values for accuracy analysis
csv_name = "chatgpt_version.csv"
df = pd.read_csv(csv_name)

# some of the sentiment cells are missing values. drop these from the data set
if csv_name == "test.csv":
    df.dropna(subset=['sentiment'], inplace=True)

# create new columns in df object to write calculated sentiment to
df["vader"] = None
df["textblob"] = None
df["textblob number"] = None
df["vader number"] = None

# need instance of sentiment analyzer
sia = SentimentIntensityAnalyzer()

# preprocess text
df["text"] = df["text"].apply(PreprocessText)

# iterate through text and collect a polarity score
# using vader and textblob
for index, row in df.iterrows():
    entry = row['text']

    # vader
    score_vader = sia.polarity_scores(entry) 
    df.at[index, "vader number"] = score_vader.get('compound')

    # textblob
    score_textblob = TextBlob(entry).sentiment.polarity 
    df.at[index, "textblob number"] = score_textblob


# calculate mean and standard deviation of the polarity scores
vader_mean = np.mean(df['vader number'])
vader_std = np.std(df['vader number'])
textblob_mean = np.mean(df['textblob number'])
textblob_std = np.std(df['textblob number'])


# use mean and std to determine cutoffs for POS/NEG/NEUT sentiment label
for index, row in df.iterrows():
    entry = row['text']

    df.at[index, "vader"] = SentCutOff(vader_mean,vader_std,df.at[index,"vader number"]) # convert numerical score to POS/NEG/NEUT and write to "vader" column

    # repeat for textblob
    df.at[index, "textblob"] = SentCutOff(textblob_mean,textblob_std,df.at[index,"textblob number"])


# calculate classification report and confusion matrix for vader and textblob results
print("Compare Vader with true sentiment.")
print(classification_report(df['sentiment'],df['vader']))
print(confusion_matrix(df['sentiment'], df['vader']))


print("Compare TextBlob with true sentiment.")
print(classification_report(df['sentiment'],df['textblob']))
print(confusion_matrix(df['sentiment'], df['textblob']))

if csv_name == "chatgpt_version.csv":
    print("Compare ChatGPT with true sentiment.")
    print(classification_report(df['sentiment'],df['chatgpt']))
    print(confusion_matrix(df['sentiment'], df['chatgpt']))