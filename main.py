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
tweets = pd.read_csv("twitter_training.csv")

# create new columns in df object to write calculated sentiment to
tweets["vader"] = None
tweets["textblob"] = None
tweets["textblob number"] = None
tweets["vader number"] = None

# need instance of sentiment analyzer
sia = SentimentIntensityAnalyzer()

# preprocess text
tweets["entry"] = tweets["entry"].apply(PreprocessText)

# iterate through text and collect a polarity score
# using vader and textblob
for index, row in df.iterrows():
    entry = row['entry']

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
    entry = row['entry']

    df.at[index, "vader"] = SentCutOff(vader_mean,vader_std,df.at[index,"vader number"]) # convert numerical score to POS/NEG/NEUT and write to "vader" column

    # repeat for textblob
    df.at[index, "textblob"] = SentCutOff(textblob_mean,textblob_std,df.at[index,"textblob number"])


# calculate classification report and confusion matrix for vader, textblob, chatgpt, and unwrap.ai results
print("Compare Vader with annotated_sentiment.")
print(classification_report(annot['annotated_sentiment'],df['vader']))
print(confusion_matrix(annot['annotated_sentiment'], df['vader']))


print("Compare TextBlob with annotated_sentiment.")
print(classification_report(annot['annotated_sentiment'],df['textblob']))
print(confusion_matrix(annot['annotated_sentiment'], df['textblob']))