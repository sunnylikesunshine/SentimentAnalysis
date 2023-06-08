# function to preprocess text

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

def PreprocessText(text):

    # tokenize the text
    tokens = word_tokenize(text.lower())

    # remove everything except for whitespace and letters
    text = re.sub(r'[^\w\s!]', '', text)

    # remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]

    # lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join tokens back into a string
    processed_text = " ".join(lemmatized_tokens)

    return processed_text