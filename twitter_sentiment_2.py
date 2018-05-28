import tweepy
from textblob import TextBlob
#French adaptor
from textblob_fr import PatternTagger, PatternAnalyzer

import numpy as np
import operator


# Step 1 - Authenticate
consumer_key= 'L3LDRqZxGarzl2lyyfr3gtCoi'
consumer_secret= 'nSG61DUfWumyrUlLHOPWfYh7k9ReEdIXKq0uTaiK5ZdRkktCGq'

access_token='38867352-qCxvKzQd5JMFNdw66wNKsa1pA7tewN0bwJAX5eLrs'
access_token_secret='gumfqsftsNwYIJGLNvmuaTtBO4OOA3MKyzuJf5V2F5YOg'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 2 - Prepare query features

#List of candidates to French Republicans Primary Elections
candidates_names = ['Sarkozy', 'Kosciusko', 'Cope', 'Juppe', 'Fillon', 'Le Maire', 'Poisson']
#Hashtag related to the debate
name_of_debate = "PrimaireLeDebat" 
#Date of the debate : October 13th
since_date = "2016-10-13"
until_date = "2016-10-14"

#Step 2b - Function of labelisation of analysis
def get_label(analysis, threshold = 0):
	if analysis.sentiment[0]>threshold:
		return 'Positive'
	else:
		return 'Negative'


#Step 3 - Retrieve Tweets and Save Them
all_polarities = dict()
for candidate in candidates_names:
	this_candidate_polarities = []
	#Get the tweets about the debate and the candidate between the dates
	this_candidate_tweets = api.search(q=[name_of_debate, candidate], count=100, since = since_date, until=until_date)
	print("candidatename: ",candidate , "count: ",this_candidate_tweets.count )
	#Save the tweets in csv
	with open('%s_tweets.csv' % candidate, 'w') as this_candidate_file:
		print('Inside when')
		this_candidate_file.write('tweet,sentiment_label\n')
		print('Before for',this_candidate_tweets[0].tweet.print)
		for tweet in this_candidate_tweets:
			analysis = TextBlob(tweet.text)
			print('After query for')
			#Get the label corresponding to the sentiment analysis
			print("Sentiment=: ",analysis.sentiment[0] )
			this_candidate_polarities.append(analysis.sentiment[0])
			this_candidate_file.write('%s,%s\n' % (tweet.text.encode('utf8'), get_label(analysis)))
	
	#Save the mean for final results
	all_polarities[candidate] = np.mean(this_candidate_polarities)
 
#Step bonus - Print a Result
sorted_analysis = sorted(all_polarities.items(), key=operator.itemgetter(1), reverse=True)
print('Mean Sentiment Polarity in descending order :')
for candidate, polarity in sorted_analysis:
	print('%s : %0.3f' % (candidate, polarity))