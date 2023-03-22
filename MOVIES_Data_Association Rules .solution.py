# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 00:22:03 2022

@author: MAHESH
"""
import pandas as pd
# pip install mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

Movie_data = pd.read_csv("my_movies.csv")
Movie_data
list(Movie_data)
Movie_data.info()

Movie_data1 = Movie_data.drop(['V1','V2','V3','V4','V5'],axis=1)
Movie_data1
list(Movie_data1)

Movie_data1.info()

# CHECKING FOR THE NULL VALUES 

Movie_data.isnull().sum()

# IMPORTING APRIORI 

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# apriori algorithm

movie_items = apriori(Movie_data1, min_support=0.1, use_colnames=True)
movie_items

confi_rules = association_rules(movie_items, metric="confidence", min_threshold=0.5)
confi_rules

rules1 = association_rules(movie_items, metric="lift", min_threshold=0.8)
rules1

rules1.sort_values('lift',ascending = False)[0:20]
rules1.sort_values('confidence',ascending = False)[0:10]

R1=rules1[rules1.lift>1]
R1

l1=R1.pivot('antecedents','consequents','lift')
l1.head()

l1=R1.pivot('antecedents','consequents','support')
l1.head()

# <<<<< EXPLORATION DATA ANALYSIS <<<<<

import matplotlib.pyplot as plt
import seaborn as sns

# histogram

rules1[['support','confidence']].hist()
rules1[['support','confidence','lift']].hist()

# box plot

rules1[['support','confidence']].boxplot()
rules1[['support','confidence','lift']].boxplot()

# scatter plot

plt.scatter(rules1['support'], rules1['confidence'])
plt.show()

# sns scatter plot 
sns.scatterplot('support','confidence', data=rules1, hue='antecedents')
plt.show()

# bar graph 

plt.bar(rules1['lift'], rules1['confidence'])
plt.show()

plt.bar(rules1['support'], rules1['lift'])
plt.show()

# TRYING WITH DIFFERENT SUPPORT VALUES,confidence values and lift values

movie_items0 = apriori(Movie_data1, min_support=0.18, use_colnames=True)
movie_items0

confi_rules = association_rules(movie_items0, metric="confidence", min_threshold=0.9)
confi_rules

rules2 = association_rules(movie_items0, metric="lift", min_threshold=0.95)
rules2

rules2.sort_values('lift',ascending = False)[0:20]
rules2.sort_values('confidence',ascending = False)[0:10]

R2=rules2[rules2.lift>1]
R2

l2=R2.pivot('antecedents','consequents','lift')
l2.head()

l2=R2.pivot('antecedents','consequents','support')
l2.head()

# <<<<< EXPLORATION DATA ANALYSIS <<<<<

import matplotlib.pyplot as plt
import seaborn as sns

# histogram

rules2[['support','confidence']].hist()
rules2[['support','confidence','lift']].hist()

# scatter plot

plt.scatter(rules2['support'], rules2['confidence'])
plt.show()

# box plot

rules2[['support','confidence']].boxplot()
rules2[['support','confidence','lift']].boxplot()

#

sns.scatterplot('support','confidence', data=rules2, hue='antecedents')
plt.show()

# bar graph 

plt.bar(rules2['lift'], rules2['confidence'])
plt.show()

plt.bar(rules2['support'], rules2['lift'])
plt.show()


