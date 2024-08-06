#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd


# In[3]:


documents = [
    "I like to play football",
    "Football is the best sport",
    "I enjoy playing football with my friends"
]


# In[4]:


# Bag-of-Words (CountVectorizer)
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(documents)


# In[5]:


# Convert to DataFrame for better visualization
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
print("Bag-of-Words (CountVectorizer):")
print(bow_df)


# In[6]:


# TF/IDF (TfidfVectorizer)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)


# In[7]:


# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF/IDF (TfidfVectorizer):")
print(tfidf_df)


# In[ ]:




