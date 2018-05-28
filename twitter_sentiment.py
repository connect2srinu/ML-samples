import tweepy
from textblob import TextBlob
import numpy as np


# Step 1 - Authenticate
consumer_key= 'L3LDRqZxGarzl2lyyfr3gtCoi'
consumer_secret= 'nSG61DUfWumyrUlLHOPWfYh7k9ReEdIXKq0uTaiK5ZdRkktCGq'

access_token='38867352-qCxvKzQd5JMFNdw66wNKsa1pA7tewN0bwJAX5eLrs'
access_token_secret='gumfqsftsNwYIJGLNvmuaTtBO4OOA3MKyzuJf5V2F5YOg'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)



#Step 3 - Retrieve Tweets
public_tweets = api.search('#RoyalWedding')



#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself
arr = []
#type1 = np.dtype([str, str])
arr = np.empty((0,2), str)
# arr = np.array(['tweet','senti'])
# a = np.array(['tweet','Sentiment']) 
# np.append(a,['sample','positive'])
for tweet in public_tweets:
    print(tweet.text)
    #tweet.tweet
    #Step 4 Perform Sentiment Analysis on Tweets
    print(arr)
    analysis = TextBlob(tweet.text)
    print([analysis.string, ('negative' ,'positive')[analysis.sentiment.polarity >= 0]])
    #arr.append([tweet.text.encode('utf8'), ('negative' ,'positive')[analysis.sentiment.polarity >= 0]])
    arr = np.append(arr,[analysis.string, ('negative' ,'positive')[analysis.sentiment.polarity >= 0]],axis=0)
    print("lang--:",analysis.detect_language)
    print(('negative' ,'positive')[analysis.sentiment.polarity >= 0], analysis.sentiment.polarity)
    print("")
#n = arr.reshape(16,2)
np_arr = np.array(arr)
print(np_arr.shape)
np.savetxt('data.csv', np_arr,delimiter=',', fmt="%s")