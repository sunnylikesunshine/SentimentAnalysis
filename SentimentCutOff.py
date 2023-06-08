# takes in list of sentiment scores and assigns POSTIVE, NEUTRAL, or NEGATIVE to each one
# sentiment scores range from -1 to 1; use mean and std of data set to determine cut-offs
# cut-offs are -1\leq NEG\leq mean, mean<NEUT<std, std\leq POS\leq 1

def SentCutOff(mean,std,score):

    if score >= std:
        sentiment = "POSITIVE"
    elif score <= mean:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment