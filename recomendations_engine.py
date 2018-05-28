import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm.datasets import fetch_stackexchange
from lightfm import LightFM

#CHALLENGE part 1 of 3 - write your own fetch and format method for a different recommendation
#dataset. Here a good few https://gist.github.com/entaroadun/1653794 
#And take a look at the fetch_movielens method to see what it's doing 
#

data_stack =fetch_stackexchange('crossvalidated',min_training_interactions=3 )
print(repr(data_stack['train']))
print(repr(data_stack['test']))

#fetch data and format it
data = fetch_movielens(min_rating=4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))
print("*****************************")
print(data)
print("*****************************")
print(data_stack)
print("*****************************")

# #CHALLENGE part 2 of 3 - use 3 different loss functions (so 3 different models), compare results, print results for
# #the best one. - Available loss functions are warp, logistic, bpr, and warp-kos.

#create model
model = LightFM(loss='warp')
model_stack_warp = LightFM(loss='warp')
model_stack_bpr = LightFM(loss='bpr')
model_stack_log = LightFM(loss='logistic')
#train model
model.fit(data['train'], epochs=30, num_threads=2)
model_stack_warp.fit(data_stack['train'], epochs=30, num_threads=2)
model_stack_bpr.fit(data_stack['train'], epochs=30, num_threads=2)
model_stack_log.fit(data_stack['train'], epochs=30, num_threads=2)


# #CHALLENGE part 3 of 3 - Modify this function so that it parses your dataset correctly to retrieve
# #the necessary variables (products, songs, tv shows, etc.)
# #then print out the recommended results 

def sample_recommendation(model, data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        known_positives = data['item_feature_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_feature_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
            
#sample_recommendation(model, data, [3, 25, 450])
print("***************************** model_stack_warp *****************************")
sample_recommendation(model_stack_warp, data_stack, [3, 25, 450])
print("***************************** model_stack_bpr *****************************")
sample_recommendation(model_stack_bpr, data_stack, [3, 25, 450])
print("***************************** model_stack_log *****************************")
sample_recommendation(model_stack_log, data_stack, [3, 25, 450])
print("***************************** data_stack info *****************************")
