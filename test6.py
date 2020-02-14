import pandas as pd
import numpy as np

from pyAgrum.lib.bn2graph import pdfize
import csv
from pandas.api.types import is_string_dtype
from math import *
import os
import math
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
from pyAgrum.lib.bn2roc import showROC
#from IPython.display import display, HTML
from ipywidgets import *
from metakernel.display import display
from metakernel.display import HTML
from eli5.sklearn import PermutationImportance


data=pd.read_csv('C:/Users/LENOVO/Desktop/gestion des risques/insurance_claim0s.csv')

# Pret pour GIT
# let's check whether the data has any null values or not.

# but there are '?' in the datset which we have to remove by NaN Values
data = data.replace('?',np.NaN)

# missing value treatment using fillna

# we will replace the '?' by the most common collision type as we are unaware of the type.
data['collision_type'].fillna(data['collision_type'].mode()[0], inplace = True)

# It may be the case that there are no responses for property damage then we might take it as No property damage.
data['property_damage'].fillna('NO', inplace = True)

# again, if there are no responses fpr police report available then we might take it as No report available
data['police_report_available'].fillna('NO', inplace = True)
"""
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# let's extrat days, month and year from policy bind date
data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'], errors = 'coerce')
data['incident_date'] = pd.to_datetime(data['incident_date'], errors = 'coerce')

# extracting days and month from date
data['incident_month'] = data['incident_date'].dt.month
data['incident_day'] = data['incident_date'].dt.day

# let's encode the fraud report to numerical values
data['fraud_reported'] = data['fraud_reported'].replace(('Y','N'),(0,1))


for y in data:
    if is_string_dtype(data[y]):
        tab=(data[[y,'fraud_reported']].groupby([y], as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False))
        data[y] = data[y].replace((tuple(str(x)for x in tab[y])),
                       (tuple((round(x,2))for x in tab['fraud_reported'])))        
# let's delete unnecassary columns
data = data.drop(['umbrella_limit','policy_number','policy_bind_date', 'incident_date','incident_location','auto_model'], axis = 1)
"""        

# let's check the columns after deleting the columns

data.to_csv('post_data.csv', index=False)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------







template=gum.BayesNet()
for k in data.keys():
    template.add(gum.LabelizedVariable(str(k), str(k), [str(x) for x in data[k].unique()]))
    
    
   
    
#gnb.showBN(template)


file = 'post_data.csv'
learner = gum.BNLearner(file, template)
bn = learner.learnBN()

gnb.showBN(bn)

learner=gum.BNLearner(os.path.join("out","sample_asia.csv"))
learner.use3off2()
learner.useNML()
ge3off2=learner.learnMixedStructure()
ge3off2

#gnb.showInference(bn)
#gnb.showInformation(bn,{},size="20")
"""
pdfize(bn, "test3")
gnb.showInference(bn,size="10", evs={'incident_severity':'Total Loss', 'incident_type':'Single Vehicle Collision'})

from imblearn.ensemble import BalancedRandomForestClassifier 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# let's split the data into dependent and independent sets

x = data.drop(['fraud_reported'], axis = 1)
y = data['fraud_reported']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
model = BalancedRandomForestClassifier(n_estimators = 100, random_state = 0)
model.fit(x_train, y_train)
from eli5.sklearn import PermutationImportance


perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())

import shap

# Seaborn visualization library
import seaborn as sns
# Create the default pairplot
sns.pairplot(data)

"""
