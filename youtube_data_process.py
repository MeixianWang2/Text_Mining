import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# #youtube_data process
# youtube_data = pd.read_csv('/Users/annawang/icloud/Documents/Twitter copy/video_result.csv')
#
# plt.figure()
# hist1,edges1 = np.histogram(youtube_data.viewCount)
# plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])
#
# plt.scatter(youtube_data.viewCount,youtube_data.likeCount)
#
#
# y = youtube_data.likeCount
# X = youtube_data.viewCount
# X = sm.add_constant(X)
#
# lr_model = sm.OLS(y,X).fit()
# # print(lr_model.summary())
# # draw a linear regression line
# X_prime = np.linspace(X.viewCount.min(), X.viewCount.max(),100)
# X_prime = sm.add_constant(X_prime)
#
# y_hat = lr_model.predict(X_prime)
# plt.scatter(X.viewCount,y)
# plt.xlabel("View Count")
# plt.ylabel("Like Count")
# plt.plot(X_prime[:,1],y_hat)
# plt.show()

#
# #twitter_data process
# twitter_data = pd.read_csv('/Users/annawang/icloud/Documents/Twitter copy/github/tweet/result.csv')
#
# plt.figure()
# hist1,edges1 = np.histogram(twitter_data.friends)
# plt.bar(edges1[:-1],hist1,width=edges1[1:]-edges1[:-1])
#
# plt.scatter(twitter_data.followers,twitter_data.retwc)
# print(plt.show())
#
# y = twitter_data.followers
# X = twitter_data.retwc
# X = sm.add_constant(X)
#
# lr_model = sm.OLS(y,X).fit()
# print(lr_model.summary())

#twitter sentiment analysis
twitter_data = pd.read_csv('/Users/annawang/icloud/Documents/Twitter copy/results_olympics.csv')

print(twitter_data.corr())
twitter_data_subjective = twitter_data[twitter_data['subjectivity']>0.5]
plt.scatter(twitter_data.retwc,twitter_data.subjectivity)
plt.scatter(twitter_data.retwc, twitter_data.polarity)
print(twitter_data_subjective.corr())
print(plt.show())

y = twitter_data.subjectivity
X = twitter_data.retwc
X = sm.add_constant(X)

lr_model = sm.OLS(y,X).fit()
print(lr_model.summary())