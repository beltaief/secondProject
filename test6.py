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


