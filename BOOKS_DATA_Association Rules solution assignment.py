# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 23:24:31 2022

@author: MAHESH
"""

import pandas as pd
# pip install mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

Book_data = pd.read_csv("book.csv")
Book_data

list(Book_data)
Book_data.info()             # list of the variable names with data set

# CHECKING FOR THE NULL VALUES 

Book_data.isnull().sum()     # finding missing values
Book_data.shape
Book_data.columns

# IMPORTING APRIORI 

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# apriori algorithm

Books_items = apriori(Book_data, min_support=0.1, use_colnames=True)
Books_items

confi_rules = association_rules(Books_items, metric="confidence", min_threshold=0.5)
confi_rules

rules1 = association_rules(Books_items, metric="lift", min_threshold=0.8)
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

Books_items0 = apriori(Book_data, min_support=0.18, use_colnames=True)
Books_items0

confi_rules = association_rules(Books_items0, metric="confidence", min_threshold=0.9)
confi_rules

rules2 = association_rules(Books_items0, metric="lift", min_threshold=0.95)
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

# sns scatter plot 

sns.scatterplot('support','confidence', data=rules2, hue='antecedents')
plt.show()

# bar graph 

plt.bar(rules2['lift'], rules2['confidence'])
plt.show()

plt.bar(rules2['support'], rules2['lift'])
plt.show()





