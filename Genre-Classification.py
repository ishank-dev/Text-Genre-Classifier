#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# In[4]:


meta = pd.read_csv("./MovieSummaries/movie.metadata.tsv", sep = '\t', header = None)
meta.head()


# In[5]:


meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]


# In[7]:


plots = []

with open("./MovieSummaries/plot_summaries.txt", 'r') as f:
       reader = csv.reader(f, dialect='excel-tab') 
       for row in tqdm(reader):
            plots.append(row)


# In[8]:


movie_id = []
plot = []

# extract movie Ids and plot summaries
for i in tqdm(plots):
  movie_id.append(i[0])
  plot.append(i[1])

# create dataframe
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})


# In[9]:


movies.head()


# In[10]:


# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)

# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

movies.head()


# In[11]:


movies['genre'][0]


# In[12]:


type(json.loads(movies['genre'][0]))


# In[13]:


json.loads(movies['genre'][0]).values()


# In[14]:


dict_values(['Drama', 'World cinema'])


# In[15]:


# an empty list
genres = [] 

# extract genres
for i in movies['genre']: 
  genres.append(list(json.loads(i).values())) 

# add to 'movies' dataframe  
movies['genre_new'] = genres


# In[16]:


movies_new = movies[~(movies['genre_new'].str.len() == 0)]


# In[17]:


movies_new.shape, movies.shape


# In[18]:


movies.head()


# In[29]:


all_genre = sum(genres,[])
len(set(all_genre))


# In[33]:



all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})


# In[34]:


g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()


# In[35]:


# function for text cleaning 
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text


# In[36]:


movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))


# In[37]:


pd.options.mode.chained_assignment = None  # default='warn'


# In[38]:


movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))


# In[39]:


def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# print 100 most frequent words 
freq_words(movies_new['clean_plot'], 100)


# In[40]:


nltk.download('stopwords')


# In[41]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))


# In[42]:


freq_words(movies_new['clean_plot'], 100)


# In[71]:


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])

# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])
print(y)


# In[44]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)


# In[45]:


xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.2, random_state=9)


# In[46]:


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)


# In[47]:


from sklearn.linear_model import LogisticRegression

# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score


# In[48]:


lr = LogisticRegression()
clf = OneVsRestClassifier(lr)


# In[183]:


# fit model on train data
clf.fit(xtrain_tfidf, ytrain)


# In[185]:


# make predictions for validation set
y_pred = clf.predict(xval_tfidf)


# In[186]:


y_pred[3]


# In[187]:


multilabel_binarizer.inverse_transform(y_pred)[3]


# In[188]:


# evaluate performance
f1_score(yval, y_pred, average="micro")


# In[189]:


y_pred_prob = clf.predict_proba(xval_tfidf)


# In[190]:


t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)


# In[191]:


# evaluate performance
f1_score(yval, y_pred_new, average="micro")


# In[192]:


def infer_tags(q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


# In[201]:


for i in range(5): 
  k = xval.sample(1).index[0]
  print("Movie: ", movies_new['movie_name'][k], "\nPredicted genre: ", infer_tags(xval[k])), print("Actual genre: ",movies_new['genre_new'][k], "\n")


# In[194]:


df1 = pd.read_csv('./IMDB-Movie-Data.csv')
df1.head()


# In[206]:


# Testing our model on IMDB dataset
for index, row in df1.iterrows():
    print(row["Title"]+'------------->'+row["Genre"])
    print('Predicted-->',infer_tags(row['Description']))
    print()


# In[ ]:





# In[ ]:





# In[ ]:




