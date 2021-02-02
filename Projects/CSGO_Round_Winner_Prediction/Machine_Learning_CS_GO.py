#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Machine-Learning-Group-Project" data-toc-modified-id="Machine-Learning-Group-Project-1">Machine Learning Group Project</a></span><ul class="toc-item"><li><span><a href="#Team-B" data-toc-modified-id="Team-B-1.1">Team B</a></span></li></ul></li><li><span><a href="#Table-of-Contents" data-toc-modified-id="Table-of-Contents-2">Table of Contents</a></span></li><li><span><a href="#0.-Load-Libraries" data-toc-modified-id="0.-Load-Libraries-3">0. Load Libraries</a></span></li><li><span><a href="#1.-Select-data" data-toc-modified-id="1.-Select-data-4">1. Select data</a></span><ul class="toc-item"><li><span><a href="#1.1.-Load-train-and-test-data-from-CSV" data-toc-modified-id="1.1.-Load-train-and-test-data-from-CSV-4.1">1.1. Load train and test data from CSV</a></span></li><li><span><a href="#1.2.-EDA" data-toc-modified-id="1.2.-EDA-4.2">1.2. EDA</a></span><ul class="toc-item"><li><span><a href="#1.2.1-Explore-general-characteristics-of-the-datasets" data-toc-modified-id="1.2.1-Explore-general-characteristics-of-the-datasets-4.2.1">1.2.1 Explore general characteristics of the datasets</a></span><ul class="toc-item"><li><span><a href="#1.2.1.1.-Train-set" data-toc-modified-id="1.2.1.1.-Train-set-4.2.1.1">1.2.1.1. Train set</a></span></li><li><span><a href="#1.2.1.2.-Test-set" data-toc-modified-id="1.2.1.2.-Test-set-4.2.1.2">1.2.1.2. Test set</a></span></li></ul></li><li><span><a href="#1.2.2.-EDA-using-ProfileReport-on-train-data" data-toc-modified-id="1.2.2.-EDA-using-ProfileReport-on-train-data-4.2.2">1.2.2. EDA using ProfileReport on train data</a></span><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-4.2.2.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li><li><span><a href="#2.-Pre-process-data" data-toc-modified-id="2.-Pre-process-data-5">2. Pre-process data</a></span><ul class="toc-item"><li><span><a href="#2.1-Outliers-identification-and-exclusion" data-toc-modified-id="2.1-Outliers-identification-and-exclusion-5.1">2.1 Outliers identification and exclusion</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-5.1.0.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li><li><span><a href="#3.-Transform-data" data-toc-modified-id="3.-Transform-data-6">3. Transform data</a></span><ul class="toc-item"><li><span><a href="#3.1.-Convert-columns-types" data-toc-modified-id="3.1.-Convert-columns-types-6.1">3.1. Convert columns types</a></span></li><li><span><a href="#3.2.-Feature-Engineering" data-toc-modified-id="3.2.-Feature-Engineering-6.2">3.2. Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#3.2.1.-Feature-creation" data-toc-modified-id="3.2.1.-Feature-creation-6.2.1">3.2.1. Feature creation</a></span></li><li><span><a href="#3.2.2.-Exploring-the-new-features" data-toc-modified-id="3.2.2.-Exploring-the-new-features-6.2.2">3.2.2. Exploring the new features</a></span></li><li><span><a href="#3.2.3.-PCA-analysis" data-toc-modified-id="3.2.3.-PCA-analysis-6.2.3">3.2.3. PCA analysis</a></span><ul class="toc-item"><li><span><a href="#3.2.3.1.-Perform-PCA-on-weapon-variables-(Standardised)" data-toc-modified-id="3.2.3.1.-Perform-PCA-on-weapon-variables-(Standardised)-6.2.3.1">3.2.3.1. Perform PCA on weapon variables (Standardised)</a></span></li><li><span><a href="#3.2.3.2.-Perform-PCA-on-all-variables-(Standardised)" data-toc-modified-id="3.2.3.2.-Perform-PCA-on-all-variables-(Standardised)-6.2.3.2">3.2.3.2. Perform PCA on all variables (Standardised)</a></span></li><li><span><a href="#3.2.3.3.-Perform-PCA-on-all-ct-and-t-variables-separated-(Standardised)" data-toc-modified-id="3.2.3.3.-Perform-PCA-on-all-ct-and-t-variables-separated-(Standardised)-6.2.3.3">3.2.3.3. Perform PCA on all ct and t variables separated (Standardised)</a></span></li><li><span><a href="#3.2.3.4.-Perform-PCA-on-pairs-of-ct-and-t-(Standardised)" data-toc-modified-id="3.2.3.4.-Perform-PCA-on-pairs-of-ct-and-t-(Standardised)-6.2.3.4">3.2.3.4. Perform PCA on pairs of ct and t (Standardised)</a></span></li><li><span><a href="#3.2.3.5.-Perform-PCA-on-delta-of-weapon-variables-(standardised)" data-toc-modified-id="3.2.3.5.-Perform-PCA-on-delta-of-weapon-variables-(standardised)-6.2.3.5">3.2.3.5. Perform PCA on delta of weapon variables (standardised)</a></span></li><li><span><a href="#3.2.3.6.-Perform-PCA-on-delta-of-weapon-variables-+-variables-without-delta-(standardised)" data-toc-modified-id="3.2.3.6.-Perform-PCA-on-delta-of-weapon-variables-+-variables-without-delta-(standardised)-6.2.3.6">3.2.3.6. Perform PCA on delta of weapon variables + variables without delta (standardised)</a></span></li><li><span><a href="#3.2.3.7.-Perform-PCA-on-weapon-variables-grouped-by-category-(standardised)" data-toc-modified-id="3.2.3.7.-Perform-PCA-on-weapon-variables-grouped-by-category-(standardised)-6.2.3.7">3.2.3.7. Perform PCA on weapon variables grouped by category (standardised)</a></span></li><li><span><a href="#3.2.3.8.-Perform-PCA-on-weapon-variables-grouped-by-category-+-variables-not-grouped-(standardised)" data-toc-modified-id="3.2.3.8.-Perform-PCA-on-weapon-variables-grouped-by-category-+-variables-not-grouped-(standardised)-6.2.3.8">3.2.3.8. Perform PCA on weapon variables grouped by category + variables not grouped (standardised)</a></span></li><li><span><a href="#3.2.3.9.-Perform-PCA-on-pairs-of-ct-and-t-for-weapon-variables-grouped-by-category-for-ct-columns-and-t-columns-(stadardised)" data-toc-modified-id="3.2.3.9.-Perform-PCA-on-pairs-of-ct-and-t-for-weapon-variables-grouped-by-category-for-ct-columns-and-t-columns-(stadardised)-6.2.3.9">3.2.3.9. Perform PCA on pairs of ct and t for weapon variables grouped by category for ct columns and t columns (stadardised)</a></span></li><li><span><a href="#3.2.3.10.-Perform-PCA-on-delta-of-weapon-variables-grouped-by-category--+--variables-not-grouped-(Standardised)" data-toc-modified-id="3.2.3.10.-Perform-PCA-on-delta-of-weapon-variables-grouped-by-category--+--variables-not-grouped-(Standardised)-6.2.3.10">3.2.3.10. Perform PCA on delta of weapon variables grouped by category  +  variables not grouped (Standardised)</a></span></li><li><span><a href="#3.2.3.11.-Perform-PCA-from-time-left-to-players-alive-(19-variables)-and-for-the-grenades-related-variables" data-toc-modified-id="3.2.3.11.-Perform-PCA-from-time-left-to-players-alive-(19-variables)-and-for-the-grenades-related-variables-6.2.3.11">3.2.3.11. Perform PCA from time left to players alive (19 variables) and for the grenades related variables</a></span></li></ul></li><li><span><a href="#3.2.4.-Transform-categorical-variables-into-dummy" data-toc-modified-id="3.2.4.-Transform-categorical-variables-into-dummy-6.2.4">3.2.4. Transform categorical variables into dummy</a></span></li><li><span><a href="#3.2.5.-Output-of-3.-Transformation" data-toc-modified-id="3.2.5.-Output-of-3.-Transformation-6.2.5">3.2.5. Output of 3. Transformation</a></span><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-6.2.5.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li><li><span><a href="#4.-Model-data" data-toc-modified-id="4.-Model-data-7">4. Model data</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#4.0.-Create-a-baseline" data-toc-modified-id="4.0.-Create-a-baseline-7.0.1">4.0. Create a baseline</a></span><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-7.0.1.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li><li><span><a href="#4.1.-Testing-dataset:-df-=-original-dataset" data-toc-modified-id="4.1.-Testing-dataset:-df-=-original-dataset-7.1">4.1. Testing dataset: df = original dataset</a></span><ul class="toc-item"><li><span><a href="#4.1.1.-Logistic-regression-model" data-toc-modified-id="4.1.1.-Logistic-regression-model-7.1.1">4.1.1. Logistic regression model</a></span><ul class="toc-item"><li><span><a href="#4.1.1.1.-Pure-logistic-regression" data-toc-modified-id="4.1.1.1.-Pure-logistic-regression-7.1.1.1">4.1.1.1. Pure logistic regression</a></span></li><li><span><a href="#4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization" data-toc-modified-id="4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization-7.1.1.2">4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization</a></span></li><li><span><a href="#4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**" data-toc-modified-id="4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**-7.1.1.3">4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**</a></span></li></ul></li><li><span><a href="#4.1.2.-Decision-Tree-Classifier-models" data-toc-modified-id="4.1.2.-Decision-Tree-Classifier-models-7.1.2">4.1.2. Decision Tree Classifier models</a></span><ul class="toc-item"><li><span><a href="#4.1.2.1.-One-decision-tree" data-toc-modified-id="4.1.2.1.-One-decision-tree-7.1.2.1">4.1.2.1. One decision tree</a></span></li><li><span><a href="#4.1.2.2.-Bagging" data-toc-modified-id="4.1.2.2.-Bagging-7.1.2.2">4.1.2.2. Bagging</a></span></li><li><span><a href="#4.1.2.3.-Random-Forest-Classifier-with-default-parameters" data-toc-modified-id="4.1.2.3.-Random-Forest-Classifier-with-default-parameters-7.1.2.3">4.1.2.3. Random Forest Classifier with default parameters</a></span></li><li><span><a href="#Get-back" data-toc-modified-id="Get-back-7.1.2.4"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li><li><span><a href="#4.2.-Testing-dataset:-df-=-delta_df" data-toc-modified-id="4.2.-Testing-dataset:-df-=-delta_df-7.2">4.2. Testing dataset: df = delta_df</a></span><ul class="toc-item"><li><span><a href="#4.2.1.-Logistic-regression-model" data-toc-modified-id="4.2.1.-Logistic-regression-model-7.2.1">4.2.1. Logistic regression model</a></span><ul class="toc-item"><li><span><a href="#4.2.1.1.-Pure-logistic-regression" data-toc-modified-id="4.2.1.1.-Pure-logistic-regression-7.2.1.1">4.2.1.1. Pure logistic regression</a></span></li><li><span><a href="#4.2.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization" data-toc-modified-id="4.2.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization-7.2.1.2">4.2.1.2. Logistic regression with cross-validation and 'Ridge' regularization</a></span></li><li><span><a href="#4.2.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**" data-toc-modified-id="4.2.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**-7.2.1.3">4.2.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**</a></span></li></ul></li><li><span><a href="#4.2.2.-Decision-Tree-Classifier-models" data-toc-modified-id="4.2.2.-Decision-Tree-Classifier-models-7.2.2">4.2.2. Decision Tree Classifier models</a></span><ul class="toc-item"><li><span><a href="#4.2.2.1.-One-decision-tree" data-toc-modified-id="4.2.2.1.-One-decision-tree-7.2.2.1">4.2.2.1. One decision tree</a></span></li><li><span><a href="#4.2.2.2.-Bagging" data-toc-modified-id="4.2.2.2.-Bagging-7.2.2.2">4.2.2.2. Bagging</a></span></li><li><span><a href="#4.2.2.3.-Random-Forest-Classifier" data-toc-modified-id="4.2.2.3.-Random-Forest-Classifier-7.2.2.3">4.2.2.3. Random Forest Classifier</a></span></li><li><span><a href="#4.2.4.-XGBoost-model" data-toc-modified-id="4.2.4.-XGBoost-model-7.2.2.4">4.2.4. XGBoost model</a></span></li></ul></li><li><span><a href="#4.2.4.-Support-Vector-Machines-(SVM)-model" data-toc-modified-id="4.2.4.-Support-Vector-Machines-(SVM)-model-7.2.3">4.2.4. Support Vector Machines (SVM) model</a></span><ul class="toc-item"><li><span><a href="#4.2.4.1.-SVM-+-RandomizedSearchCV-+-kernel=linear" data-toc-modified-id="4.2.4.1.-SVM-+-RandomizedSearchCV-+-kernel=linear-7.2.3.1">4.2.4.1. SVM + RandomizedSearchCV + kernel=linear</a></span></li><li><span><a href="#4.2.4.2.-SVM-+-RandomizedSearchCV-+-kernel=rbf" data-toc-modified-id="4.2.4.2.-SVM-+-RandomizedSearchCV-+-kernel=rbf-7.2.3.2">4.2.4.2. SVM + RandomizedSearchCV + kernel=rbf</a></span></li><li><span><a href="#4.2.4.3.-SVM-+-RandomizedSearchCV-+-kernel=poly" data-toc-modified-id="4.2.4.3.-SVM-+-RandomizedSearchCV-+-kernel=poly-7.2.3.3">4.2.4.3. SVM + RandomizedSearchCV + kernel=poly</a></span></li><li><span><a href="#Get-back" data-toc-modified-id="Get-back-7.2.3.4"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li><li><span><a href="#4.3-Testing-dataset:-df-=-weapcat_deltaa_df" data-toc-modified-id="4.3-Testing-dataset:-df-=-weapcat_deltaa_df-7.3">4.3 Testing dataset: df = weapcat_deltaa_df</a></span><ul class="toc-item"><li><span><a href="#4.1.1.-Logistic-regression-model" data-toc-modified-id="4.1.1.-Logistic-regression-model-7.3.1">4.1.1. Logistic regression model</a></span><ul class="toc-item"><li><span><a href="#4.1.1.1.-Pure-logistic-regression" data-toc-modified-id="4.1.1.1.-Pure-logistic-regression-7.3.1.1">4.1.1.1. Pure logistic regression</a></span></li><li><span><a href="#4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization" data-toc-modified-id="4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization-7.3.1.2">4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization</a></span></li><li><span><a href="#4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**" data-toc-modified-id="4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**-7.3.1.3">4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**</a></span></li></ul></li><li><span><a href="#4.1.2.-Decision-Tree-Classifier-models" data-toc-modified-id="4.1.2.-Decision-Tree-Classifier-models-7.3.2">4.1.2. Decision Tree Classifier models</a></span><ul class="toc-item"><li><span><a href="#4.1.2.1.-One-decision-tree" data-toc-modified-id="4.1.2.1.-One-decision-tree-7.3.2.1">4.1.2.1. One decision tree</a></span></li><li><span><a href="#4.1.2.2.-Bagging" data-toc-modified-id="4.1.2.2.-Bagging-7.3.2.2">4.1.2.2. Bagging</a></span></li><li><span><a href="#4.1.2.3.-Random-Forest-Classifier-with-default-parameters" data-toc-modified-id="4.1.2.3.-Random-Forest-Classifier-with-default-parameters-7.3.2.3">4.1.2.3. Random Forest Classifier with default parameters</a></span></li><li><span><a href="#Get-back" data-toc-modified-id="Get-back-7.3.2.4"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li><li><span><a href="#4.4-Testing-dataset:-df-=-pca_weapcat_deltaa_df" data-toc-modified-id="4.4-Testing-dataset:-df-=-pca_weapcat_deltaa_df-7.4">4.4 Testing dataset: df = pca_weapcat_deltaa_df</a></span><ul class="toc-item"><li><span><a href="#4.1.1.-Logistic-regression-model" data-toc-modified-id="4.1.1.-Logistic-regression-model-7.4.1">4.1.1. Logistic regression model</a></span><ul class="toc-item"><li><span><a href="#4.1.1.1.-Pure-logistic-regression" data-toc-modified-id="4.1.1.1.-Pure-logistic-regression-7.4.1.1">4.1.1.1. Pure logistic regression</a></span></li><li><span><a href="#4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization" data-toc-modified-id="4.1.1.2.-Logistic-regression-with-cross-validation-and-'Ridge'-regularization-7.4.1.2">4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization</a></span></li><li><span><a href="#4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**" data-toc-modified-id="4.1.1.3.-Logistic-regression-with-cross-validation-and-'Lasso'-Regularization**-7.4.1.3">4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**</a></span></li></ul></li><li><span><a href="#4.1.2.-Decision-Tree-Classifier-models" data-toc-modified-id="4.1.2.-Decision-Tree-Classifier-models-7.4.2">4.1.2. Decision Tree Classifier models</a></span><ul class="toc-item"><li><span><a href="#4.1.2.1.-One-decision-tree" data-toc-modified-id="4.1.2.1.-One-decision-tree-7.4.2.1">4.1.2.1. One decision tree</a></span></li><li><span><a href="#4.1.2.2.-Bagging" data-toc-modified-id="4.1.2.2.-Bagging-7.4.2.2">4.1.2.2. Bagging</a></span></li><li><span><a href="#4.1.2.3.-Random-Forest-Classifier-with-default-parameters" data-toc-modified-id="4.1.2.3.-Random-Forest-Classifier-with-default-parameters-7.4.2.3">4.1.2.3. Random Forest Classifier with default parameters</a></span></li><li><span><a href="#Get-back" data-toc-modified-id="Get-back-7.4.2.4"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li><li><span><a href="#5.-Best-model---training" data-toc-modified-id="5.-Best-model---training-8">5. Best model - training</a></span><ul class="toc-item"><li><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-8.0.0.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li><li><span><a href="#6.-Final-model---test" data-toc-modified-id="6.-Final-model---test-9">6. Final model - test</a></span><ul class="toc-item"><li><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Get-back" data-toc-modified-id="Get-back-9.0.0.1"><a href="#tc">Get back</a></a></span></li></ul></li></ul></li></ul></li></ul></div>

# # Machine Learning Group Project
# ## Team B

# In[1]:


# Set the display
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>")) # Increase cell width
display(HTML("<style>.rendered_html { font-size: 16px; }</style>")) # Increase font size


# To tackle the exercise we will follow the **Machine Learning pipeline** as stated in class:

# In[1]:


from IPython.display import Image
Image(filename='ML_process.png')


# <a id='tc'></a>

# # Table of Contents
# 
# * [1. Select data](#1)
# * [2. Pre-process data](#2)
# * [3. Transform data ](#3)
#     * [3.1. Outliers identification and exclusion](#3.1.)
#     * [3.2. Feature Engineering](#3.2.)
#         * [3.2.1. Feature creationg](#3.2.1.)
#         * [3.2.2. Exploring the new features](#3.2.2.)
#         * [3.2.3. PCA analysis](#3.2.3.)
#         * [3.2.4. Transform categorical variables into dummy](#3.2.4.)
# * [4. Model data](#4)
# * [5. Best Model - training](#5)
# * [6. Final model - test](#6)

# # 0. Load Libraries

# In[1]:


# Import the necessary libraries for the analysis

# Data processing
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Scikit libraries
from sklearn import linear_model, datasets, svm, metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, plot_confusion_matrix, mean_squared_error, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder 
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, truncnorm, randint
import xgboost

# For plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from pprint import pprint


# # 1. Select data
# <a id='1'></a>

# We were provided with 2 datasets: 
# 
# * training_set.csv
# * test_set.csv
# 
# Both were part of the Kaggle competition: https://www.kaggle.com/christianlillelund/csgo-round-winner-classification
# 
# The dataset consists of ~700 demos from high level tournament play in 2019 and 2020. Warmup rounds and restarts have been filtered, and for the remaining live rounds a round snapshot have been recorded every 20 seconds until the round is decided. Following its initial publication, It has been pre-processed and flattened to improve readability and make it easier for algorithms to process. The total number of snapshots is 122411.

# ## 1.1. Load train and test data from CSV

# In[2]:


# Loading the csv files

df = pd.read_csv('training_set.csv')
testdf = pd.read_csv('test_set.csv')


# In[3]:


df.head(5)


# ## 1.2. EDA
# <a id='1.2'></a>

# ### 1.2.1 Explore general characteristics of the datasets

# #### 1.2.1.1. Train set

# In[4]:


df.shape


# In[5]:


# To identify the types
df.dtypes


# #### 1.2.1.2. Test set

# In[6]:


testdf.shape


# In[7]:


# To identify the types
testdf.dtypes


# ### 1.2.2. EDA using ProfileReport on train data

# In[8]:


# Getting a detailed report of the dataset, as a first 

# reporttrain = ProfileReport(df)


# In[9]:


# As we have many features, is better to export the report not to crash 'google chrome'

# reporttrain.to_file("CS_GO_Report.html")


# Transform these 2 variables from categorical to numerical, we run these here for technical reasons of the code, but they are repeated in the Tranfrom Data section because there it's their proper place.

# In[10]:


# Converting column 'bomb_planted' into a numeric
df['bomb_planted'] = df['bomb_planted'].astype('float64')


# In[11]:


df['bomb_planted'].unique()


# In[12]:


# Converting column 'round_winner' into numeric
df['round_winner'].replace(['CT','T'], [0,1], inplace=True)


# In[13]:


df['round_winner'].unique()


# #### [Get back](#tc)

# # 2. Pre-process data
# <a id='2'></a>

# ## 2.1 Outliers identification and exclusion
# <a id='2.1'></a>

# In[14]:


# We organize the columns so the single columns are first and the columns that come in "tuples" stay together. e.g. all counter terrorist related column and terrorist related columns

cols_to_order = ['round_winner','time_left', 'map','bomb_planted', 'ct_defuse_kits', 'ct_grenade_flashbang', 't_grenade_flashbang']
new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
df = df[new_columns]


# In[15]:


# Create business parameter to flag outliers for each variable; if a value of the variable is higher than this parameter the entire row will be considered an outlier from a business point of view and dropped

parameters_1 = [-1, -1, -1, -1, 5, 10, 10, -1, -1, 500, 500, 500, 500, 80000, 80000] #values fir variables until money
parameters_2 = [5] * 82 #values for all the rest of weapons
parameters = parameters_1 + parameters_2
ser_par = pd.Series(parameters, index = df.columns)


# In[16]:


# Understand for each feature if it has 'outliers' and 'NAs'

results = pd.DataFrame(np.nan, index=df.columns, columns=['Outlier', 'NAs'])

    for i, j in enumerate(ser_par):
        if j > 0:
            a = (df[ser_par.index[i]]>j).sum().astype('bool')
            b = df[ser_par.index[i]].isna().sum().astype('bool')
            results.loc[ser_par.index[i], 'Outlier'] = a
            results.loc[ser_par.index[i], 'NAs'] = b  


# In[17]:


# Display which rows have outliers and NAs

results[(results['Outlier'] == True) | (results['NAs'] == True)]


# In[18]:


# Create a dataframe with the number of occurrences for each value for the weapons ralated variables

results2 = pd.DataFrame()

# Subset the columns associated with weapons, grenades and the defuse kit

for i in df.columns[np.r_[4:7, 15:len(df.columns)]]:
    x = df.groupby(i)[i].count()
    results2 = pd.concat([results2, x], axis = 1)


# Calculate the transposed matrix, table of frequencies

results3 = results2.T
results3[results3.index.isin(results[(results['Outlier'] == True) | (results['NAs'] == True)].index.tolist())]


# Now we will calculate for each row how many columns are not respecting the business rules defined in 'ser_par'.
# 
# For that we test versus the Dataframe of True/False 'results'

# In[19]:


df.head(2)


# In[20]:


ser_par


# In[21]:


# Checking vs the business rules 'ser_par'

result4 = pd.DataFrame(index = df.index)

for i, j in enumerate(ser_par):
    if j > 0:
        a = df[ser_par.index[i]] > j
        result4 = pd.concat([result4, a], axis = 1)
    else:
        b = df[ser_par.index[0]] < -1
        b = pd.Series(b, name = ser_par.index[i])
        result4 = pd.concat([result4, b], axis = 1)


# In[22]:


# Calculate sum by row to sum how many True (outlier = 1) values are present in that row

sum_row = result4.sum(axis = 1)


# In[23]:


# Here shows how many rows are going to be dropped from the original df

sum_row[sum_row > 0].count()


# In[24]:


# Here shows the index of the rows that are going to be dropped

sum_row[sum_row > 0]


# In[25]:


# These are the rows that we need to drop from the dataframe because they are not within the boundaries defined by the business rule

df.loc[sum_row > 0]


# These are the rows that we will keep in the dataframe.

# In[26]:


df = df.loc[sum_row == 0].reset_index(drop=True)
df.head(5)


# In[27]:


# Original dataframe shape               = (82014, 97)
# New dataframe without 'outliers' shape = (81999, 97)
# The difference = 15 , are the rows that had outliers in one or more columns
df.shape


# #### [Get back](#tc)

# # 3. Transform data
# <a id='3'></a>

# ## 3.1. Convert columns types

# As said before, we run these cells before, but their natural place is here.

# In[28]:


# Converting column 'bomb_planted' into a numeric
# df['bomb_planted'] = df['bomb_planted'].astype('float64')


# In[29]:


# df['bomb_planted'].unique()


# In[30]:


# Converting column 'round_winner' into numeric
# df['round_winner'].replace(['CT','T'], [0,1], inplace=True)


# In[31]:


# df['round_winner'].unique()


# ## 3.2. Feature Engineering
# <a id='3.2.'></a>

# In order to understand the impact of the different features towards predicting the target variable, we analyized the correlation between features of the same 'type' of column and their difference.
# Then we plotted everything into a Heatmap for each 'type' of columns
# 
# Conclusion: 
#    * Overall the heatmaps, show that the impact of the difference or 'delta' is more relevant to predict the target variable than the original features by themselves
#    * The delta features seems promising, but further analysis need to be done before modeling
# 
# e.g. Columns health: ct_health = -0.19, t_health = 0.09, delta_health = -0.4 // 0 = counter terrorist, 1 = terrorist

# ### 3.2.1. Feature creation
# <a id='3.2.1.'></a>

# In[32]:


# First we subset the first columns of the dataset to run the profile

df_1 = df.copy() # avoid changing the original daaframe by mak
corr_df = pd.DataFrame(columns = ['Var_name','Target_variable'])
delta_df = pd.DataFrame(df.iloc[:,0:5])

i = 5

# we create the delta columns
while i < (len(df.columns)):
    
    a = df_1.iloc[:,i]
    b = df_1.iloc[:,i+1]
    df_1['delta_'+df_1.iloc[:,i].name[3:]] = a - b
    c = df_1['round_winner']
    
    
    delta_df['delta_'+df_1.iloc[:,i].name[3:]] = a - b
    corr = df_1[[a.name, b.name,'delta_'+df_1.iloc[:,i].name[3:]]].corrwith(c)
    corr_df = corr_df.append({'Var_name':a.name,'Target_variable':corr[0]}, ignore_index=True)
    corr_df = corr_df.append({'Var_name':b.name,'Target_variable':corr[1]}, ignore_index=True)
    corr_df = corr_df.append({'Var_name':'delta_'+df_1.iloc[:,i].name[3:],'Target_variable':corr[2]}, ignore_index=True)

    corr = df_1[[a.name, b.name,'delta_'+df_1.iloc[:,i].name[3:],'round_winner']].corr()
    sns.heatmap(corr, annot = True)
    plt.title('Columns '+df_1.iloc[:,i].name[3:])
    plt.show()
    i = i + 2

del(corr)


# We are going to create some columns based on the weapon category. We will follow the next steps:
# 
#    1. Obtain a weapons dataframe, with the categories of each weapon of the game
#    2. Create 2 new columns corresponding to the names of the columns in the pre-processed 'df', to match coincidences
#    3. Create 4 additional dataframes:
#        
#        3.1 showing the sum of weapons per category for both ct and t: "weapcat_sum_df"
#       
#        3.2 showing the avg of weapons per category for both ct and t: "weapcat_avg_df"
#        
#        3.2 showing the difference per weapon category between ct and t: "weapcat_deltas_df"
#        
#        3.4 showing the difference of the average between ct and t per weapon category: "weapcat_deltaa_df"

# In[33]:


# 1. we import the dataframe containing the weapons category description

cat_df = pd.read_csv('WeaponCategory.csv')


# In[34]:


# 2. Create 2 new columns corresponding to the names of the columns in df, so we can have a correspondence 

cat_df['ct_cols'] = "ct_" + cat_df['Weapon_name'] 
cat_df['t_cols'] = "t_" + cat_df['Weapon_name'] 


# In[35]:


# 3. create 4 dataframes: 
#   3.1 showing the sum of weapons per category for both ct and t
#   3.2 showing the avg of weapons per category for both ct and t
#   3.3 showing the difference of the sums between ct and t per weapon category  
#   3.4 showing the difference of the average between ct and t per weapon category  

weapcat_sum_df = pd.DataFrame(df.iloc[:,np.r_[0:5, 7:19]])
weapcat_avg_df = pd.DataFrame(df.iloc[:,np.r_[0:5, 7:19]])
weapcat_deltas_df = pd.DataFrame(delta_df.iloc[:,np.r_[0:5, 6:12]])
weapcat_deltaa_df = pd.DataFrame(delta_df.iloc[:,np.r_[0:5, 6:12]])

for i in cat_df['Category'].unique():
    #print(i)
    ct = df[cat_df[cat_df['Category']==i]['ct_cols']]
    t = df[cat_df[cat_df['Category']==i]['t_cols']]
    
    weapcat_sum_df['ct_'+i] = ct.sum(axis=1)
    weapcat_sum_df['t_'+i] = t.sum(axis=1)
    weapcat_avg_df['ct_'+i] = ct.mean(axis=1)
    weapcat_avg_df['t_'+i] = t.mean(axis=1)
    
    weapcat_deltas_df['dlt_'+i] = ct.sum(axis=1) -t.sum(axis=1)
    weapcat_deltaa_df['dlt_'+i] = ct.mean(axis=1) -t.mean(axis=1)
    
del(cat_df,ct, t)


# In[36]:


weapcat_deltaa_df.head()


# ### 3.2.2. Exploring the new features
# <a id='3.2.2'></a>
# 
# Understand the relations of the different features with the target variable

# **df dataframe**

# In[67]:


# Utilize the 'mutual_info_classif' to obtain information about predictivity power of the feature regarding the target variable

mi = mutual_info_classif(df.drop(["round_winner", "map"], axis=1), df.round_winner)
mi_df = pd.DataFrame(mi,index=df.iloc[:,np.r_[1, 3:len(df.columns)]].columns, columns =['mi_score'])


# In[68]:


# Checking the results, of those features that have a mi > 5%

mi_df[mi_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# In[69]:


# Reviewing the correlations with a Matrix

f = plt.figure(figsize=(20, 20))
plt.matshow(df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(df.shape[1]), df.columns, fontsize=10, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# **delta_df dataframe**

# In[70]:


# Checking the 'mi' of the new dataframe with deltas 'mi_dlt_df', with 'mi'>5%

mi_delta = mutual_info_classif(delta_df.drop(["round_winner", "map"], axis=1), delta_df.round_winner)
mi_dlt_df = pd.DataFrame(mi_delta,index=delta_df.iloc[:,np.r_[1, 3:len(delta_df.columns)]].columns, columns =['mi_score'])
mi_dlt_df[mi_dlt_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# In[71]:


# Plot the correlations matrix of the deltas 'mi_dlt_df'

f = plt.figure(figsize=(15, 15))
plt.matshow(delta_df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(delta_df.shape[1]), delta_df.columns, fontsize=10, rotation=90)
plt.yticks(range(delta_df.shape[1]), delta_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# As we can see, the new deltas matrix has better features, in the term of predicting power

# **weapcat_sum_df dataframe**

# In[72]:


# Now we will check the 'mi' of the matrix: mi_wc_sum_df

mi_wc_sum = mutual_info_classif(weapcat_sum_df.drop(["round_winner", "map"], axis=1), weapcat_sum_df.round_winner)
mi_wc_sum_df = pd.DataFrame(mi_wc_sum,index=weapcat_sum_df.iloc[:,np.r_[1, 3:len(weapcat_sum_df.columns)]].columns, columns =['mi_score'])
mi_wc_sum_df[mi_wc_sum_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# In[73]:


# Again, plotting the correlations matrix for easy visualizing

f = plt.figure(figsize=(15, 8))
plt.matshow(weapcat_sum_df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(weapcat_sum_df.shape[1]), weapcat_sum_df.columns, fontsize=10, rotation=90)
plt.yticks(range(weapcat_sum_df.shape[1]), weapcat_sum_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# **weapcat_deltaa_df dataframe**

# In[74]:


# Same procedure to check 'mi' of the matrix: mi_wc_deltaa_df

mi_wc_deltaa = mutual_info_classif(weapcat_deltaa_df.drop(["round_winner", "map"], axis=1), weapcat_deltaa_df.round_winner)
mi_wc_deltaa_df = pd.DataFrame(mi_wc_deltaa,index=weapcat_deltaa_df.iloc[:,np.r_[1, 3:len(weapcat_deltaa_df.columns)]].columns, columns =['mi_score'])
mi_wc_deltaa_df[mi_wc_deltaa_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# In[75]:


# Visualizing the correlation matrix

f = plt.figure(figsize=(10, 8))
plt.matshow(weapcat_deltaa_df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(weapcat_deltaa_df.shape[1]), weapcat_deltaa_df.columns, fontsize=10, rotation=90)
plt.yticks(range(weapcat_deltaa_df.shape[1]), weapcat_deltaa_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# **weapcat_deltas_df dataframe**

# In[76]:


# Again with 'mi_wc_deltas_df'

mi_wc_deltas = mutual_info_classif(weapcat_deltas_df.drop(["round_winner", "map"], axis=1), weapcat_deltas_df.round_winner)
mi_wc_deltas_df = pd.DataFrame(mi_wc_deltas,index=weapcat_deltas_df.iloc[:,np.r_[1, 3:len(weapcat_deltas_df.columns)]].columns, columns =['mi_score'])
mi_wc_deltas_df[mi_wc_deltas_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# In[77]:


# Visualizing correlation matrix of 'mi_wc_deltas_df'

f = plt.figure(figsize=(10, 10))
plt.matshow(weapcat_deltas_df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(weapcat_deltas_df.shape[1]), weapcat_deltas_df.columns, fontsize=10, rotation=90)
plt.yticks(range(weapcat_deltas_df.shape[1]), weapcat_deltas_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# In[29]:


# Visualize weapcat_deltaa_df

sns.pairplot(delta_df)


# ### 3.2.3. PCA analysis
# <a id='3.2.3.'></a>
# 
# We will perform the following PCA analysis:
#     
#     3.2.3.1. Perform PCA on weapon variables (Standardised)
#     3.2.3.2. Perform PCA on all variables (Standardised)
#     3.2.3.3. Perform PCA on all ct and t variables separated (Standardised)
#     3.2.3.4. Perform PCA on pairs of ct and t (Standardised)
#     3.2.3.5. Perform PCA on delta of weapon variables (standardised)
#     3.2.3.6. Perform PCA on delta of weapon variables + variables without delta (standardised)
#     3.2.3.7. Perform PCA on weapon variables grouped by category (standardised)
#     3.2.3.8. Perform PCA on weapon variables grouped by category + variables not grouped (standardised)
#     3.2.3.9. Perform PCA on pairs of ct and t for weapon variables grouped by category for ct columns and t columns (standardised)
#     3.2.3.10. Perform PCA on delta of weapon variables grouped by category  +  variables not grouped (Standardised)
#     3.2.3.11. Perform PCA from time left to players alive (19 variables) and for the grenades related variables

# #### 3.2.3.1. Perform PCA on weapon variables (Standardised)

# In[439]:


df_input = df.iloc[:,15:] #take all weapon variables


# In[440]:


df_input.head(2)


# In[441]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[442]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[443]:


pca.fit(X) #fit the pca


# In[444]:


pca.n_components_ #show number of principal components created


# In[445]:


df_pca = pca.transform(X) #create array with PC values


# In[446]:


col = list(range(1,58)) #create columns for the df


# In[447]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[448]:


principalDf.head(2)


# In[449]:


#pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
pca.explained_variance_ratio_ 


# In[450]:


# Check the correlation between each PC and the target variable, sorting the values as absolute

principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).abs().sort_values(ascending=False).head(10)  #check the correlation between each PC and the target variable


# In[451]:


df_map_pca = pd.concat([df.iloc[:,0], principalDf.iloc[:,0:]], axis = 1)
df_map_pca.head()


# In[452]:


# Plot the pca result

f = plt.figure(figsize=(10, 10))
plt.matshow(df_map_pca.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(df_map_pca.shape[1]), df_map_pca.columns, fontsize=10, rotation=90)
plt.yticks(range(df_map_pca.shape[1]), df_map_pca.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# In[453]:


# Analyzing 'mi' with the pca results and the target variable

mi_pca = mutual_info_classif(df_map_pca.drop(["round_winner"], axis=1), df_map_pca.round_winner)
mi_pca_df = pd.DataFrame(mi_pca,index=df_map_pca.iloc[:,1:].columns, columns =['mi_score'])
mi_pca_df[mi_pca_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# #### 3.2.3.2. Perform PCA on all variables (Standardised)

# In[454]:


df_input = pd.concat([df.iloc[:,1], df.iloc[:,3:]], axis = 1)
df_input.head(2)


# In[455]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[456]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[457]:


pca.fit(X) #fit the pca


# In[458]:


pca.n_components_ #show number of principal components created


# In[459]:


df_pca = pca.transform(X) #create array with PC values


# In[460]:


col = list(range(1,62)) #create columns for the df


# In[461]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[462]:


principalDf.head(2)


# In[463]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))


# In[464]:


principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values() #check the correlation between each PC and the target variable


# In[465]:


a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.3. Perform PCA on all ct and t variables separated (Standardised)

# In[466]:


df_input = df.iloc[:,5:]
df_input.head(2)


# In[467]:


df_input_ct = df_input.iloc[:, list(range(0, len(df_input.columns), 2))]
df_input_ct.head(2)


# In[468]:


df_input_t = df_input.iloc[:, list(range(1, len(df_input.columns), 2))]
df_input_t.head(2)


# **'CT' Variables**

# In[469]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[470]:


X_temp = df_input_ct.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[471]:


pca.fit(X) #fit the pca


# In[472]:


pca.n_components_ #show number of principal components created


# In[473]:


df_pca = pca.transform(X) #create array with PC values


# In[474]:


col = list(range(1,30)) #create columns for the df


# In[475]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[476]:


principalDf.head(2)


# In[477]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))


# In[478]:


principalDf.iloc[:,0].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[479]:


a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# **'T' Variables**

# In[480]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[481]:


X_temp = df_input_t.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[482]:


pca.fit(X) #fit the pca


# In[483]:


pca.n_components_ #show number of principal components created


# In[484]:


df_pca = pca.transform(X) #create array with PC values


# In[485]:


col = list(range(1,34)) #create columns for the df


# In[486]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[487]:


principalDf.head(2)


# In[488]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))


# In[489]:


a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.4. Perform PCA on pairs of ct and t (Standardised)

# **A. Players alive**

# In[490]:


df_input = df[["ct_players_alive", "t_players_alive"]]
df_input.head(2)


# In[491]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[492]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[493]:


pca.fit(X) #fit the pca


# In[494]:


pca.n_components_ #show number of principal components created


# In[495]:


df_pca = pca.transform(X) #create array with PC values


# In[496]:


col = list(range(1,3)) #create columns for the df


# In[497]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[498]:


principalDf.head(2)


# In[499]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))
a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# **B. Health**

# In[500]:


df_input = df[["ct_health", "t_health"]]
df_input.head(2)


# In[501]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[502]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[503]:


pca.fit(X) #fit the pca


# In[504]:


pca.n_components_ #show number of principal components created


# In[505]:


df_pca = pca.transform(X) #create array with PC values


# In[506]:


col = list(range(1,3)) #create columns for the df


# In[507]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[508]:


principalDf.head(2)


# In[509]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[510]:


principalDf.iloc[:,0].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[511]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# **C. Money**

# In[512]:


df_input = df[["ct_money", "t_money"]]
df_input.head(2)


# In[513]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[514]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[515]:


pca.fit(X) #fit the pca


# In[516]:


pca.n_components_ #show number of principal components created


# In[517]:


df_pca = pca.transform(X) #create array with PC values


# In[518]:


col = list(range(1,3)) #create columns for the df


# In[519]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[520]:


principalDf.head(2)


# In[521]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[522]:


principalDf.iloc[:,1].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[523]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# **D. Armor**

# In[524]:


df_input = df[["ct_armor", "t_armor"]]
df_input.head(2)


# In[525]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[526]:


X_temp = df_input.values # We do not take the class label
X = StandardScaler().fit_transform(X_temp) # normalizing the features


# In[527]:


pca.fit(X) #fit the pca


# In[528]:


pca.n_components_ #show number of principal components created


# In[529]:


df_pca = pca.transform(X) #create array with PC values


# In[530]:


col = list(range(1,3)) #create columns for the df


# In[531]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df


# In[532]:


principalDf.head(2)


# In[533]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[534]:


principalDf.iloc[:,0].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[535]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.5. Perform PCA on delta of weapon variables (standardised)

# In[536]:


df_input = delta_df.iloc[:,np.r_[5, 10:len(delta_df.columns)]]
df_input.head(2)


# In[537]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[538]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[539]:


pca.fit(X_std) #fit the pca


# In[540]:


pca.n_components_ #show number of principal components created


# In[541]:


df_pca = pca.transform(X_std) #create array with PC values


# In[542]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df


# In[543]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[544]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[545]:


principalDf.iloc[:,2].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[546]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.6. Perform PCA on delta of weapon variables + variables without delta (standardised)

# In[547]:


df_input = pd.concat([delta_df.iloc[:,1], delta_df.iloc[:,3:]], axis = 1)
df_input.head(2)


# In[548]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[549]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[550]:


pca.fit(X_std) #fit the pca


# In[551]:


pca.n_components_ #show number of principal components created


# In[552]:


df_pca = pca.transform(X_std) #create array with PC values


# In[553]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df
#col


# In[554]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[555]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[556]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.7. Perform PCA on weapon variables grouped by category (standardised)

# In[557]:


df_input = weapcat_sum_df.iloc[:,17:]
df_input.head(2)


# In[558]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[559]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[560]:


pca.fit(X_std) #fit the pca


# In[561]:


pca.n_components_ #show number of principal components created


# In[562]:


df_pca = pca.transform(X_std) #create array with PC values


# In[563]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df
#col


# In[564]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[565]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[566]:


principalDf.iloc[:,2].corr(df.iloc[:, 0]) #check the correlation between each PC and the target variable


# In[567]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.8. Perform PCA on weapon variables grouped by category + variables not grouped (standardised)

# In[568]:


df_input = pd.concat([weapcat_sum_df.iloc[:,1], weapcat_sum_df.iloc[:,3:]], axis = 1)
df_input.head(2)


# In[569]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[570]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[571]:


pca.fit(X_std) #fit the pca


# In[572]:


pca.n_components_ #show number of principal components created


# In[573]:


df_pca = pca.transform(X_std) #create array with PC values


# In[574]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df


# In[575]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[576]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[577]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.9. Perform PCA on pairs of ct and t for weapon variables grouped by category for ct columns and t columns (stadardised)

# In[578]:


df_input = weapcat_sum_df.iloc[:,17:]
df_input.head(2)


# **CT columns**

# In[579]:


df_input_ct = df_input.iloc[:, list(range(0, len(df_input.columns), 2))]
df_input_ct.head(2)


# In[580]:


X = df_input_ct.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[581]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[582]:


pca.fit(X_std) #fit the pca


# In[583]:


pca.n_components_ #show number of principal components created


# In[584]:


df_pca = pca.transform(X_std) #create array with PC values


# In[585]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df


# In[586]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[587]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[588]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,0:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# **T columns**

# In[589]:


df_input_t = df_input.iloc[:, list(range(1, len(df_input.columns), 2))]
df_input_t.head(2)


# In[590]:


X = df_input_t.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[591]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[592]:


pca.fit(X_std) #fit the pca


# In[593]:


pca.n_components_ #show number of principal components created


# In[594]:


df_pca = pca.transform(X_std) #create array with PC values


# In[595]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df
#col


# In[596]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[597]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[598]:


pca.explained_variance_ratio_.sort() #show the percentage of the variance explained by each PC
print('variance expl: ' + str(round(min(pca.explained_variance_ratio_),5))+ ' - ' +str(round(max(pca.explained_variance_ratio_),5)))

a =  principalDf.iloc[:,0:].corrwith(df.iloc[:, 0]).sort_values()
print('correlation: ' + str(round(min(a),5))+ ' - ' +str(round(max(a),5)))
del(a)


# #### 3.2.3.10. Perform PCA on delta of weapon variables grouped by category  +  variables not grouped (Standardised)

# In[599]:


df_input = pd.concat([weapcat_deltaa_df.iloc[:,1], weapcat_deltaa_df.iloc[:,3:]], axis = 1)
df_input.head(2)


# In[600]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[601]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[602]:


pca.fit(X_std) #fit the pca


# In[603]:


pca.n_components_ #show number of principal components created


# In[604]:


df_pca = pca.transform(X_std) #create array with PC values


# In[605]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df
#col


# In[606]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[607]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[608]:


pca_weapcat_deltaa_df = pd.concat([weapcat_deltaa_df.iloc[:,0],weapcat_deltaa_df.iloc[:,2], principalDf.iloc[:,0:]], axis = 1)
pca_weapcat_deltaa_df.head()


# In[208]:


pca_weapcat_deltaa_df.iloc[:,0:].corrwith(weapcat_deltaa_df.iloc[:, 0]).abs().sort_values(ascending=False) #check the correlation between each PC and the target variable


# In[209]:


f = plt.figure(figsize=(10, 10))
plt.matshow(pca_weapcat_deltaa_df.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(pca_weapcat_deltaa_df.shape[1]), pca_weapcat_deltaa_df.columns, fontsize=10, rotation=90)
plt.yticks(range(pca_weapcat_deltaa_df.shape[1]), pca_weapcat_deltaa_df.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# In[210]:


mi_pca = mutual_info_classif(df_map_pca.drop(["round_winner"], axis=1), df_map_pca.round_winner)
mi_pca_df = pd.DataFrame(mi_pca,index=df_map_pca.iloc[:,1:].columns, columns =['mi_score'])
mi_pca_df[mi_pca_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# #### 3.2.3.11. Perform PCA from time left to players alive (19 variables) and for the grenades related variables

# In[211]:


df_input = pd.concat([df.iloc[:,1], df.iloc[:,3:18], df.iloc[:,-10:-1]], axis = 1)
#df_input.head(2)
df_input.shape


# In[212]:


X = df_input.values # We do not take the class label
X_std = StandardScaler().fit_transform(X) # normalizing the features


# In[213]:


pca = PCA(0.9) #define percentage of variance that we want to explain


# In[214]:


pca.fit(X_std) #fit the pca


# In[215]:


pca.n_components_ #show number of principal components created


# In[216]:


df_pca = pca.transform(X_std) #create array with PC values


# In[217]:


col = list(range(1,df_pca.shape[1]+1)) #create columns for the df


# In[218]:


principalDf = pd.DataFrame(data = df_pca
             , columns = col) #put the values of PC into a df
principalDf.head(2)


# In[219]:


pca.explained_variance_ratio_ #show the percentage of the variance explained by each PC


# In[220]:


df.iloc[:,1:].corrwith(df.iloc[:, 0]).sort_values() #check the correlation between each PC and the target variable


# In[221]:


principalDf.iloc[:,1:].corrwith(df.iloc[:, 0]).abs().sort_values(ascending=False).sort_values(ascending=False) #check the correlation between each PC and the target variable


# In[222]:


df_map_pca = pd.concat([df.iloc[:,0], principalDf.iloc[:,0:]], axis = 1)
df_map_pca.head()


# In[223]:


f = plt.figure(figsize=(10, 10))
plt.matshow(df_map_pca.corr(), fignum=f.number, cmap=plt.cm.coolwarm)
plt.xticks(range(df_map_pca.shape[1]), df_map_pca.columns, fontsize=10, rotation=90)
plt.yticks(range(df_map_pca.shape[1]), df_map_pca.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)


# In[224]:


mi_pca = mutual_info_classif(df_map_pca.drop(["round_winner"], axis=1), df_map_pca.round_winner)
mi_pca_df = pd.DataFrame(mi_pca,index=df_map_pca.iloc[:,1:].columns, columns =['mi_score'])
mi_pca_df[mi_pca_df['mi_score']>0.05].sort_values(by='mi_score', ascending=False)


# ### 3.2.4. Transform categorical variables into dummy
# <a id='3.2.4.'></a>

# In order to transform categorical variables into dummy variables we used 'one-hot enconder'

# In[225]:


# In order to transform categorical variables into dummy variables we used 'one-hot enconder'
# Create an one-hot encoder for map variables in the df dataframe

ohe = OneHotEncoder()

df['map'] = df.map.astype('category')
categorical_cols = df.columns[df.dtypes=='category'].tolist()
hot_enc = pd.DataFrame(ohe.fit_transform(df[categorical_cols]).toarray(), columns=ohe.get_feature_names(categorical_cols)).reset_index(drop=True)

df = pd.concat([df, hot_enc], axis = 1)
df = df.drop(['map'], axis=1)

df.head(2)


# In[43]:


# Create an one-hot encoder for map variables in the delta_df dataframe

ohe = OneHotEncoder()

delta_df['map'] = delta_df.map.astype('category')
categorical_cols = delta_df.columns[delta_df.dtypes=='category'].tolist()
hot_enc = pd.DataFrame(ohe.fit_transform(delta_df[categorical_cols]).toarray(), columns=ohe.get_feature_names(categorical_cols)).reset_index(drop=True)

delta_df = pd.concat([delta_df, hot_enc], axis = 1)
delta_df = delta_df.drop(['map'], axis=1)

delta_df.head(2)


# In[394]:


weapcat_deltaa_df.head()


# In[609]:


# Create an one-hot encoder for map variables in the weapcat_deltaa_df dataframe

ohe = OneHotEncoder()

weapcat_deltaa_df['map'] = weapcat_deltaa_df.map.astype('category')
categorical_cols = weapcat_deltaa_df.columns[weapcat_deltaa_df.dtypes=='category'].tolist()
hot_enc = pd.DataFrame(ohe.fit_transform(weapcat_deltaa_df[categorical_cols]).toarray(), columns=ohe.get_feature_names(categorical_cols)).reset_index(drop=True)

weapcat_deltaa_df = pd.concat([weapcat_deltaa_df, hot_enc], axis = 1)
weapcat_deltaa_df = weapcat_deltaa_df.drop(['map'], axis=1)

weapcat_deltaa_df.head(2)


# In[610]:


# Create an one-hot encoder for map variables in the pca_weapcat_deltaa_df dataframe

ohe = OneHotEncoder()

pca_weapcat_deltaa_df['map'] = pca_weapcat_deltaa_df.map.astype('category')
categorical_cols = pca_weapcat_deltaa_df.columns[pca_weapcat_deltaa_df.dtypes=='category'].tolist()
hot_enc = pd.DataFrame(ohe.fit_transform(pca_weapcat_deltaa_df[categorical_cols]).toarray(), columns=ohe.get_feature_names(categorical_cols)).reset_index(drop=True)

pca_weapcat_deltaa_df = pd.concat([pca_weapcat_deltaa_df, hot_enc], axis = 1)
pca_weapcat_deltaa_df = pca_weapcat_deltaa_df.drop(['map'], axis=1)


# ### 3.2.5. Output of 3. Transformation
# <a id='3.2.5.'></a>
# 
# After the different analysis and transformations performed, we obtained 4 main dataset to test with the different models:
# 
# - **Original one 'df'**: All the original columns with map column with hot-encoder (Number of features = 104)
#   
# - **Deltas 'delta_df'**: Columns round winner,  time_left, map (encoded), bomb planted, ct_defuse_kits. Plus the delta columns (Number of features = 59)
#   
# - **Variable groups by category of guns 'weapcat_deltaa_df'**: Takes the weapon and grenade related variables, grouped them by category (rifle, pistol, etc), calculate de average and then calculate the difference between ct and t columns. Concatenated with the variables that are not weapon related (Number of features = 24)
#   
# - **Weapons category grouped + deltas + PCA 'pca_weapcat_deltaa_df'**: Standarize the weapcat_deltaa_df dataframe and run PCA analysis (Number of features = 18)

# **Original 'df'**

# In[229]:


df.shape


# In[230]:


df.head()


# **Deltas 'delta_df'**

# In[231]:


delta_df.shape


# In[232]:


delta_df.head()


# **Variable groups by category of guns 'weapcat_deltaa_df'**

# In[233]:


weapcat_deltaa_df.shape


# In[234]:


weapcat_deltaa_df.head()


# **Weapons category grouped + deltas + PCA 'pca_weapcat_deltaa_df'**

# In[235]:


pca_weapcat_deltaa_df.shape


# In[236]:


pca_weapcat_deltaa_df.head()


# #### [Get back](#tc)

# # 4. Model data
# <a id='4'></a>

# ### 4.0. Create a baseline

# In[630]:


weapcat_deltaa_df2 = weapcat_deltaa_df.copy()


# In[314]:


weapcat_deltaa_df.groupby(['round_winner'])['round_winner'].count()


# In[315]:


weapcat_deltaa_df['armor_bins'] = pd.cut(weapcat_deltaa_df['delta_armor'], bins=[-500,-400,-300,-200,-100,0,100,200,300,400,500], labels=['-500','-400','-300','-200','-100','0','100','200','300','400']).astype('category')

armor_1 = weapcat_deltaa_df[weapcat_deltaa_df['delta_armor']<100].groupby(['armor_bins','round_winner'])['armor_bins'].count().unstack()
armor = pd.DataFrame(armor_1, index = armor_1.index)
armor.rename(columns={0: 'ct', 1:'t'}, inplace=True)


# In[316]:


weapcat_deltaa_df['money_bins'] = pd.cut(weapcat_deltaa_df['delta_money'], bins=[-70000,-60000,-50000,-40000,-30000,-20000,-10000,0,10000,20000,30000,40000,50000,60000,70000], labels=[-70000,-60000,-50000,-40000,-30000,-20000,-10000,0,10000,20000,30000,40000,50000,60000]).astype('category')

money_1 = weapcat_deltaa_df[weapcat_deltaa_df['delta_armor']<100].groupby(['money_bins','round_winner'])['delta_money'].count().unstack()
money = pd.DataFrame(money_1, index = money_1.index)
money.rename(columns={0: 'ct', 1:'t'}, inplace=True)


# In[317]:


weapcat_deltaa_df['time_bin'] = pd.cut(weapcat_deltaa_df['time_left'], bins=[-5,15,35,55,75,95,115,135,155,175], labels=[15,35,55,75,95,115,135,155,175]).astype('category')

time_left_1 = weapcat_deltaa_df[(weapcat_deltaa_df['delta_armor']<100) & (weapcat_deltaa_df['delta_money']>0)].groupby(['time_bin','round_winner'])['time_bin'].count().unstack()
time_left = pd.DataFrame(time_left_1, index = time_left_1.index)
time_left.rename(columns={0: 'ct', 1:'t'}, inplace=True)


# In[318]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (25, 3))

ax1.plot( armor.index, 'ct', data=armor,  color='darkorchid', linewidth=2)
ax1.plot( armor.index, 't', data=armor,  color='steelblue',  linewidth=2)
ax1.set_title('delta_armor')
ax1.axvline(x = 5, ymin=0, ymax=15000, color='mediumvioletred')
ax1.legend()

# if delta_armor > 100 then round winner = ct

ax2.plot( money.index, 'ct', data=money,  color='darkorchid', linewidth=2)
ax2.plot( money.index, 't', data=money,  color='steelblue', linewidth=2)
ax2.set_title('delta_money')
ax2.vlines(x=0, ymin=0, ymax=17500, color='cyan')

ax2.legend()

ax3.plot( time_left.index, 'ct', data=time_left,  color='darkorchid', linewidth=2)
ax3.plot( time_left.index, 't', data=time_left,  color='steelblue', linewidth=2)
ax3.set_title('time_left')
ax3.legend()


#if delta_money<0 then round_winner = 'tw'


# In[319]:


threshold = 0

row_indexes=weapcat_deltaa_df[(weapcat_deltaa_df['delta_armor']>=threshold)].index
weapcat_deltaa_df.loc[row_indexes,'rw_predicted']=0

row_indexes=weapcat_deltaa_df[(weapcat_deltaa_df['delta_armor']<threshold)].index
weapcat_deltaa_df.loc[row_indexes,'rw_predicted']=1


# In[320]:


#The accuracy of the model is basically the total number of correct predictions divided by total number of predictions. 
weapcat_deltaa_df['predicted_test'] =  np.where(weapcat_deltaa_df['round_winner'] == weapcat_deltaa_df['rw_predicted'], True, False)
accuracy = weapcat_deltaa_df['predicted_test'].sum()/weapcat_deltaa_df['predicted_test'].count()
accuracy


# #### [Get back](#tc)

# ## 4.1. Testing dataset: df = original dataset
# <a id='4.1.'></a>

# In[104]:


# Uncomment the desired dataset for modelling:

df_mod = df.copy()   
# df_mod = delta_df.copy()
# df_mod = weapcat_deltaa_df.copy()
# df_mod = pca_weapcat_deltaa_df.copy()


# In[1]:


# Define the function to split the dataset into train-test 
def split_df(dataframe, seed=None, percentage=0.7):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return train_test_split(X, y, test_size=1-percentage, random_state=42)


# In[106]:


# Before starting we split the dataset

X_train, X_test, y_train, y_test = split_df(df_mod)


# In[107]:


# Define the function to split the dataset into train-test 

def split_df_x_y(dataframe):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return X, y


# In[108]:


# Before starting we split the dataset

X, y = split_df_x_y(df_mod)


# We split the dataframe in X and y and in 'train' and 'test' because some algorithms do not provide the validation score.
# For those algorithms and methods that do provide the validation score we will use the full dataset (extracting the validation score)

# ### 4.1.1. Logistic regression model

# #### 4.1.1.1. Pure logistic regression

# In[46]:


model = linear_model.LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_train, y_train) #fit log regression on train data


# In[47]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,model.predict(X_test))))


# #### 4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization

# In[48]:


# Ridge

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

ridge_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l2',Cs=alphas, n_jobs=6).fit(X_train, y_train)


# In[49]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,ridge_mod_cv.predict(X_test))))
plot_confusion_matrix(ridge_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# #### 4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**

# In[244]:


# Lasso

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

lasso_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l1',solver='liblinear',Cs=alphas, n_jobs = 6, verbose = 1).fit(X_train, y_train)


# In[245]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,lasso_mod_cv.predict(X_test))))
plot_confusion_matrix(lasso_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# ### 4.1.2. Decision Tree Classifier models

# #### 4.1.2.1. One decision tree

# In[52]:


df_treec = DecisionTreeClassifier(random_state=42)
df_treec.fit(X_train, y_train)


# In[53]:


predictions = df_treec.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,df_treec.predict(X_test))))
plot_confusion_matrix(df_treec, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# In[54]:


# Casting the columns into string before ploting

X_train.columns = X_train.columns.map(str)


# In[57]:


# Plotting the feature importance

plt.figure(figsize=(15,20))
plt.barh(X_train.columns, df_treec.feature_importances_)
plt.title('Feature Importance', fontsize=16);


# Now we will try to improve the accuracy fine-tuning the hyperparameters with GridSearch

# In[59]:


param_grid = {'max_depth': range(1,60)}

df_treec_pruned_cv = GridSearchCV(df_treec, 
                   param_grid,
                   scoring='accuracy',
                   cv=10 , n_jobs= 6, verbose=1)

df_treec_pruned_cv.fit(X,y)
print("Best score found on development set:")
print()
print(df_treec_pruned_cv.best_score_)
print("Best parameters set found on development set:")
print()
print(df_treec_pruned_cv.best_params_)
print()
print("Grid scores on development set:")
print()
means = df_treec_pruned_cv.cv_results_['mean_test_score']
stds = df_treec_pruned_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, df_treec_pruned_cv.cv_results_['params']):
    print("Accuracy = %0.3f (+/%0.03f) for %r" % (mean, std * 2, params))


# In[62]:


means


# In[63]:


plt.figure(figsize=(15,15))
plt.errorbar(range(1,60,1), [-m for m in means], yerr=stds, fmt='-o')
plt.title('Accuracy for different Depths', fontsize=20)
plt.xlabel("Depth", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)


# #### 4.1.2.2. Bagging

# In[69]:


dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True, n_jobs = 6, verbose = 1)
bag_model=bag_model.fit(X_train,y_train)


# In[70]:


y_test_pred=bag_model.predict(X_test)

print(bag_model.score(X_test, y_test))

# print(confusion_matrix(y_test, y_test_pred)) 

#sn.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, cmap=plt.cm.Blues)


# #### 4.1.2.3. Random Forest Classifier with default parameters

# In[67]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[68]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[370]:


#Setting model parameters
#number of trees
n_estimators = [100]


# In[371]:


model_params = {
    
    'n_estimators': n_estimators
    
}


# In[372]:


clf=RandomForestClassifier(n_estimators=100)

estimator = GridSearchCV(clf, param_grid=model_params, cv=5, verbose=1, n_jobs=6, return_train_score=True, scoring = 'accuracy')

estimator.fit(X, y);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))


# #### [Get back](#tc)

# ## 4.2. Testing dataset: df = delta_df
# <a id='4.2.'></a>

# In[45]:


# Uncomment the desired dataset for modelling:

# df_mod = df.copy()   
df_mod = delta_df.copy()
# df_mod = weapcat_deltaa_df.copy()
# df_mod = pca_weapcat_deltaa_df.copy()


# In[46]:


# Define the function to split the dataset into train-test 
def split_df(dataframe, seed=None, percentage=0.7):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return train_test_split(X, y, test_size=1-percentage, random_state=42)


# In[47]:


# Before starting we split the dataset

X_train, X_test, y_train, y_test = split_df(df_mod)


# In[48]:


# Define the function to split the dataset into train-test 

def split_df_x_y(dataframe):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return X, y


# In[49]:


# Before starting we split the dataset

X, y = split_df_x_y(df_mod)


# ### 4.2.1. Logistic regression model

# #### 4.2.1.1. Pure logistic regression

# In[268]:


model = linear_model.LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_train, y_train) #fit log regression on train data


# In[269]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,model.predict(X_test))))


# #### 4.2.1.2. Logistic regression with cross-validation and 'Ridge' regularization

# In[270]:


# Ridge

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

ridge_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l2',Cs=alphas, n_jobs=6).fit(X_train, y_train)


# In[271]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,ridge_mod_cv.predict(X_test))))
plot_confusion_matrix(ridge_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# #### 4.2.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**

# In[272]:


# Lasso

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

lasso_mod_cv = linear_model.LogisticRegressionCV(max_iter=100,penalty='l1',solver='liblinear',Cs=alphas).fit(X_train, y_train)


# In[273]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,lasso_mod_cv.predict(X_test))))
plot_confusion_matrix(lasso_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# ### 4.2.2. Decision Tree Classifier models

# #### 4.2.2.1. One decision tree

# In[274]:


df_treec = DecisionTreeClassifier(random_state=42)
df_treec.fit(X_train, y_train)


# In[275]:


predictions = df_treec.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,df_treec.predict(X_test))))
plot_confusion_matrix(df_treec, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# In[276]:


# Casting the columns into string before ploting

X_train.columns = X_train.columns.map(str)


# In[277]:


# Plotting the feature importance

plt.figure(figsize=(10,10))
plt.barh(X_train.columns, df_treec.feature_importances_)
plt.title('Feature Importance', fontsize=16);


# Now we will try to improve the accuracy fine-tuning the hyperparameters with GridSearch

# In[278]:


param_grid = {'max_depth': range(1,60)}

df_treec_pruned_cv = GridSearchCV(df_treec, 
                   param_grid,
                   scoring='accuracy',
                   cv=10 , n_jobs= 6, verbose=1)

df_treec_pruned_cv.fit(X,y)
print("Best score found on development set:")
print()
print(df_treec_pruned_cv.best_score_)
print("Best parameters set found on development set:")
print()
print(df_treec_pruned_cv.best_params_)
print()
print("Grid scores on development set:")
print()
means = df_treec_pruned_cv.cv_results_['mean_test_score']
stds = df_treec_pruned_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, df_treec_pruned_cv.cv_results_['params']):
    print("Accuracy = %0.3f (+/%0.03f) for %r" % (mean, std * 2, params))


# In[281]:


plt.figure(figsize=(10,10))
plt.errorbar(range(1,60,1), [-m for m in means], yerr=stds, fmt='-o')
plt.title('Accuracy for different Depths', fontsize=20)
plt.xlabel("Depth", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)


# #### 4.2.2.2. Bagging

# In[286]:


dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True)
bag_model=bag_model.fit(X_train,y_train)


# In[287]:


y_test_pred=bag_model.predict(X_test)

print(bag_model.score(X_test, y_test))

# print(confusion_matrix(y_test, y_test_pred)) 

#sn.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, cmap=plt.cm.Blues)


# #### 4.2.2.3. Random Forest Classifier

# ##### 4.2.2.3.1. Random Forest Classifier with default parameters

# In[462]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[463]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ##### 4.2.2.3.2. Random Forest Classifier + randomized search CV

# We impute in the classifier the best combinations of parameters identified with RandomizedSearchCV on another notebook

# In[326]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(bootstrap= True, 
                           class_weight= None, 
                           criterion= 'gini', 
                           max_depth= 70,
                           max_features= 'log2',
                           max_leaf_nodes= None,
                           min_impurity_decrease= 0.0,
                           min_impurity_split= None,
                           min_samples_leaf= 1,
                           min_samples_split= 2,
                           min_weight_fraction_leaf= 0.0,
                           n_estimators= 850,
                           n_jobs=6,
                           oob_score= False,
                           random_state= None,
                           verbose=1,
                           warm_start= False)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[327]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In the main document we say that the accuracy is as of: 0.8547, while in the previous line of code is: 0.8504.
# 
# The difference arise because we trained in a different machine the algorithm with the full 'train_set' using RandomizeSearchCV and extracted the best parameters that we used on the previous cell.
# 
# The 0.8547 is the cross-validation accuracy provided by the RandomizeSearchCV using 100% of the 'train_set', while the 0.8504 has been trained on 70% of the 'train_set'

# #### 4.2.4. XGBoost model

# We impute in the classifier the best combinations of parameters identified with RandomizedSearchCV on another notebook

# In[288]:


xgb_tree = xgboost.XGBClassifier(objective="binary:logistic",
                                 random_state=42,
                                 colsample_bytree= 1.0,
                                 gamma= 0,
                                 learning_rate= 0.10255685129067764,
                                 max_depth= 8,
                                 min_child_weight= 1,
                                 n_estimators= 451,
                                 subsample= 0.6)
xgb_tree.fit(X_train, y_train)
predictions = xgb_tree.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,xgb_tree.predict(X_test))))
plot_confusion_matrix(xgb_tree, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues);


# ### 4.2.4. Support Vector Machines (SVM) model

# In[53]:


# Basic SVM with default values
SVM_mod = svm.SVC(verbose=1)
SVM_mod.fit(X_train, y_train)
predictions = SVM_mod.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,SVM_mod.predict(X_test))))
plot_confusion_matrix(SVM_mod, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues);


# As can be seen by the accuracy obtained from the 'default' SVM, is very important to fine tune the hyperparameters of this algorithm.
# 
# The problem is that it takes a lot of time to compute, below we have the code we would have loved to run but couldn't due to time constraints and lack of computational power

# #### 4.2.4.1. SVM + RandomizedSearchCV + kernel=linear

# Considering the computational power of our computers, we decided not to test all the hyperparameters of SVM and instead of using GridSearch we will use RandomizedSearch

# In[ ]:


# Using 'RandomizedSearchCV' for tuning the most relevant hyperparameters of the SVM model kernel = linear

param_grid = {"C":[0.01, 0.1, 1, 10, 100, 1000]}
n_iter_search = 20

estimator = RandomizedSearchCV(svm.SVC(kernel='linear'), param_distributions=param_grid, cv=5, n_iter=n_iter_search, verbose=1, n_jobs=6)
estimator.fit(X_train, y_train);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))

scores = estimator.cv_results_['mean_test_score'].reshape(len(param_grid['C'])))


# In[ ]:


# Draw heatmap of the validation accuracy as a function of gamma and C

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.title('Validation accuracy')
plt.show()


# #### 4.2.4.2. SVM + RandomizedSearchCV + kernel=rbf

# In[ ]:


# Using 'RandomizedSearchCV' for tuning the most relevant hyperparameters of the SVM model kernel = linear

param_grid = {"C":[0.01, 0.1, 1, 10, 100, 1000], "gamma":[0.1, 0.01, 0.001, 1, 10, 100]}
n_iter_search = 20

estimator = RandomizedSearchCV(svm.SVC(kernel='rbf'), param_distributions=param_grid, cv=5, n_iter=n_iter_search, verbose=1, n_jobs=6)
estimator.fit(X_train, y_train);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))

scores = estimator.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))


# In[ ]:


# Draw heatmap of the validation accuracy as a function of gamma and C

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.title('Validation accuracy')
plt.show()


# #### 4.2.4.3. SVM + RandomizedSearchCV + kernel=poly

# In[ ]:


# Using 'RandomizedSearchCV' for tuning the most relevant hyperparameters of the SVM model kernel = linear

param_grid = {"C":[0.01, 0.1, 1, 10, 100, 1000], "gamma":[0.1, 0.01, 0.001, 1, 10, 100]}
n_iter_search = 20

estimator = RandomizedSearchCV(svm.SVC(kernel='rbf'), param_distributions=param_grid, cv=5, n_iter=n_iter_search, verbose=1, n_jobs=6)
estimator.fit(X_train, y_train);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))

scores = estimator.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))


# In[ ]:


# Draw heatmap of the validation accuracy as a function of gamma and C

plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.Blues)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation=45)
plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
plt.title('Validation accuracy')
plt.show()


# #### [Get back](#tc)

# ## 4.3 Testing dataset: df = weapcat_deltaa_df
# <a id='4.3.'></a>

# In[621]:


# Uncomment the desired dataset for modelling:

# df_mod = df.copy()   
# df_mod = delta_df.copy()
df_mod = weapcat_deltaa_df2.copy()
# df_mod = pca_weapcat_deltaa_df.copy()


# In[622]:


# Define the function to split the dataset into train-test 
def split_df(dataframe, seed=None, percentage=0.7):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return train_test_split(X, y, test_size=1-percentage, random_state=42)


# In[623]:


# Before starting we split the dataset

X_train, X_test, y_train, y_test = split_df(df_mod)


# In[624]:


# Define the function to split the dataset into train-test 

def split_df_x_y(dataframe):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return X, y


# In[625]:


# Before starting we split the dataset

X, y = split_df_x_y(df_mod)


# We split the dataframe in X and y and in 'train' and 'test' because some algorithms do not provide the validation score.
# For those algorithms and methods that do provide the validation score we will use the full dataset (extracting the validation score)

# ### 4.1.1. Logistic regression model

# #### 4.1.1.1. Pure logistic regression

# In[296]:


model = linear_model.LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_train, y_train) #fit log regression on train data


# In[297]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,model.predict(X_test))))


# #### 4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization

# In[298]:


# Ridge

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

ridge_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l2',Cs=alphas, n_jobs=6).fit(X_train, y_train)


# In[299]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,ridge_mod_cv.predict(X_test))))
plot_confusion_matrix(ridge_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# #### 4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**

# In[244]:


# Lasso

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

lasso_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l1',solver='liblinear',Cs=alphas, n_jobs = 6, verbose = 1).fit(X_train, y_train)


# In[245]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,lasso_mod_cv.predict(X_test))))
plot_confusion_matrix(lasso_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# ### 4.1.2. Decision Tree Classifier models

# #### 4.1.2.1. One decision tree

# In[300]:


df_treec = DecisionTreeClassifier(random_state=42)
df_treec.fit(X_train, y_train)


# In[301]:


predictions = df_treec.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,df_treec.predict(X_test))))
plot_confusion_matrix(df_treec, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# In[302]:


# Casting the columns into string before ploting

X_train.columns = X_train.columns.map(str)


# In[303]:


# Plotting the feature importance

plt.figure(figsize=(15,20))
plt.barh(X_train.columns, df_treec.feature_importances_)
plt.title('Feature Importance', fontsize=16);


# Now we will try to improve the accuracy fine-tuning the hyperparameters with GridSearch

# In[304]:


param_grid = {'max_depth': range(1,60)}

df_treec_pruned_cv = GridSearchCV(df_treec, 
                   param_grid,
                   scoring='accuracy',
                   cv=10 , n_jobs= 6, verbose=1)

df_treec_pruned_cv.fit(X,y)
print("Best score found on development set:")
print()
print(df_treec_pruned_cv.best_score_)
print("Best parameters set found on development set:")
print()
print(df_treec_pruned_cv.best_params_)
print()
print("Grid scores on development set:")
print()
means = df_treec_pruned_cv.cv_results_['mean_test_score']
stds = df_treec_pruned_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, df_treec_pruned_cv.cv_results_['params']):
    print("Accuracy = %0.3f (+/%0.03f) for %r" % (mean, std * 2, params))


# In[305]:


means


# In[306]:


plt.figure(figsize=(15,15))
plt.errorbar(range(1,60,1), [-m for m in means], yerr=stds, fmt='-o')
plt.title('Accuracy for different Depths', fontsize=20)
plt.xlabel("Depth", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)


# #### 4.1.2.2. Bagging

# In[307]:


dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True, n_jobs = 6, verbose = 1)
bag_model=bag_model.fit(X_train,y_train)


# In[308]:


y_test_pred=bag_model.predict(X_test)

print(bag_model.score(X_test, y_test))

# print(confusion_matrix(y_test, y_test_pred)) 

#sn.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, cmap=plt.cm.Blues)


# #### 4.1.2.3. Random Forest Classifier with default parameters

# In[309]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[310]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[627]:


#Setting model parameters
#number of trees
n_estimators = [100]


# In[628]:


model_params = {
    
    'n_estimators': n_estimators
    
}


# In[629]:


clf=RandomForestClassifier(n_estimators=100)

estimator = GridSearchCV(clf, param_grid=model_params, cv=5, verbose=1, n_jobs=6, return_train_score=True, scoring = 'accuracy')

estimator.fit(X, y);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))


# #### [Get back](#tc)

# ## 4.4 Testing dataset: df = pca_weapcat_deltaa_df
# <a id='4.4.'></a>

# In[611]:


# Uncomment the desired dataset for modelling:

# df_mod = df.copy()   
# df_mod = delta_df.copy()
# df_mod = weapcat_deltaa_df.copy()
df_mod = pca_weapcat_deltaa_df.copy()


# In[612]:


# Define the function to split the dataset into train-test 
def split_df(dataframe, seed=None, percentage=0.7):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return train_test_split(X, y, test_size=1-percentage, random_state=42)


# In[613]:


# Before starting we split the dataset

X_train, X_test, y_train, y_test = split_df(df_mod)


# In[614]:


# Define the function to split the dataset into train-test 

def split_df_x_y(dataframe):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return X, y


# In[615]:


# Before starting we split the dataset

X, y = split_df_x_y(df_mod)


# We split the dataframe in X and y and in 'train' and 'test' because some algorithms do not provide the validation score.
# For those algorithms and methods that do provide the validation score we will use the full dataset (extracting the validation score)

# ### 4.1.1. Logistic regression model

# #### 4.1.1.1. Pure logistic regression

# In[342]:


model = linear_model.LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_train, y_train) #fit log regression on train data


# In[343]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,model.predict(X_test))))


# #### 4.1.1.2. Logistic regression with cross-validation and 'Ridge' regularization

# In[344]:


# Ridge

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

ridge_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l2',Cs=alphas, n_jobs=6).fit(X_train, y_train)


# In[345]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,ridge_mod_cv.predict(X_test))))
plot_confusion_matrix(ridge_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# #### 4.1.1.3. Logistic regression with cross-validation and 'Lasso' Regularization**

# In[346]:


# Lasso

# Setting different parameters to test
alphas = 10**np.linspace(-1,-4,100)

lasso_mod_cv = linear_model.LogisticRegressionCV(max_iter=1000,penalty='l1',solver='liblinear',Cs=alphas, n_jobs = 6, verbose = 1).fit(X_train, y_train)


# In[347]:


print("Accuracy = {0:.4f}".format(accuracy_score(y_test,lasso_mod_cv.predict(X_test))))
plot_confusion_matrix(lasso_mod_cv, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# ### 4.1.2. Decision Tree Classifier models

# #### 4.1.2.1. One decision tree

# In[348]:


df_treec = DecisionTreeClassifier(random_state=42)
df_treec.fit(X_train, y_train)


# In[349]:


predictions = df_treec.predict(X_test)

print("Accuracy = {0:.4f}".format(accuracy_score(y_test,df_treec.predict(X_test))))
plot_confusion_matrix(df_treec, X_test, y_test, y_test.unique(), cmap=plt.cm.Blues)


# In[350]:


# Casting the columns into string before ploting

X_train.columns = X_train.columns.map(str)


# In[351]:


# Plotting the feature importance

plt.figure(figsize=(15,20))
plt.barh(X_train.columns, df_treec.feature_importances_)
plt.title('Feature Importance', fontsize=16);


# Now we will try to improve the accuracy fine-tuning the hyperparameters with GridSearch

# In[352]:


param_grid = {'max_depth': range(1,60)}

df_treec_pruned_cv = GridSearchCV(df_treec, 
                   param_grid,
                   scoring='accuracy',
                   cv=10 , n_jobs= 6, verbose=1)

df_treec_pruned_cv.fit(X,y)
print("Best score found on development set:")
print()
print(df_treec_pruned_cv.best_score_)
print("Best parameters set found on development set:")
print()
print(df_treec_pruned_cv.best_params_)
print()
print("Grid scores on development set:")
print()
means = df_treec_pruned_cv.cv_results_['mean_test_score']
stds = df_treec_pruned_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, df_treec_pruned_cv.cv_results_['params']):
    print("Accuracy = %0.3f (+/%0.03f) for %r" % (mean, std * 2, params))


# In[353]:


means


# In[354]:


plt.figure(figsize=(15,15))
plt.errorbar(range(1,60,1), [-m for m in means], yerr=stds, fmt='-o')
plt.title('Accuracy for different Depths', fontsize=20)
plt.xlabel("Depth", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)


# #### 4.1.2.2. Bagging

# In[619]:


dtc = DecisionTreeClassifier(criterion="entropy")
bag_model=BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True, n_jobs = 6, verbose = 1)
bag_model=bag_model.fit(X_train,y_train)


# In[620]:


y_test_pred=bag_model.predict(X_test)

print(bag_model.score(X_test, y_test))

# print(confusion_matrix(y_test, y_test_pred)) 

#sn.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, cmap=plt.cm.Blues)


# #### 4.1.2.3. Random Forest Classifier with default parameters

# In[1]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[358]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[616]:


#Setting model parameters
#number of trees
n_estimators = [100]


# In[617]:


model_params = {
    
    'n_estimators': n_estimators
    
}


# In[618]:


clf=RandomForestClassifier(n_estimators=100)

estimator = GridSearchCV(clf, param_grid=model_params, cv=5, verbose=1, n_jobs=6, return_train_score=True, scoring = 'accuracy')

estimator.fit(X, y);

print("The best parameters are %s with a score of %0.5f"
      % (estimator.best_params_, estimator.best_score_))


# #### [Get back](#tc)

#  # 5. Best model - training
#  <a id='5'></a>

# In[90]:


# We use as final train dataset the delta_df

df_mod = delta_df.copy()


# In[91]:


# Define the function to split the dataset into train-test 

def split_df_x_y(dataframe):
    
    X = dataframe.loc[:, dataframe.columns != 'round_winner']
    y = dataframe['round_winner']

    return X, y


# In[92]:


# Before starting we split the dataset

X, y = split_df_x_y(df_mod)


# Here we train the final model with the full train set, using the best hyper parameters found in the Randomised Search we conducted.

# In[93]:


#Create a Gaussian Classifier
clf=RandomForestClassifier(bootstrap= True, 
                           class_weight= None, 
                           criterion= 'gini', 
                           max_depth= 70,
                           max_features= 'log2',
                           max_leaf_nodes= None,
                           min_impurity_decrease= 0.0,
                           min_impurity_split= None,
                           min_samples_leaf= 1,
                           min_samples_split= 2,
                           min_weight_fraction_leaf= 0.0,
                           n_estimators= 850,
                           n_jobs=6,
                           oob_score= False,
                           random_state= 42,
                           verbose=1,
                           warm_start= False)

# Fit the model with the full train set
clf.fit(X, y)


# #### [Get back](#tc)

#  # 6. Final model - test
#   <a id='6'></a>

# Once we tried several combinations of models with the different datasets we are going to optimize the one (dataset + models) that bear the best results until now.
# 
# We will then train that model with the full Dataset (without spliting) and then use that with the Test.dataset provided

# In[94]:


# upload test df again if needed
testdf = pd.read_csv('test_set.csv')


# In[95]:


testdf.head()


# In[96]:


# Converting column 'bomb_planted' into a numeric

testdf['bomb_planted'] = testdf['bomb_planted'].astype('float64')
testdf['bomb_planted'].unique()


# In[97]:


# We organize the columns so the single columns are first and the columns that come in "tuples" stay together. e.g. all counter terrorist related column and terrorist related columns

cols_to_order = ['time_left', 'map','bomb_planted',  'ct_defuse_kits', 'ct_grenade_flashbang', 't_grenade_flashbang']
new_columns = cols_to_order + (testdf.columns.drop(cols_to_order).tolist())
testdf = testdf[new_columns]


# **Feature Engineering**

# In[98]:


# create delta_df dataframe calculating the difference between the tuples for ct and t

testdf_1 = testdf.copy()
delta_testdf = pd.DataFrame(testdf.iloc[:,0:4])

i = 4

# we create the delta columns
while i < (len(testdf.columns)):
    
    a = testdf_1.iloc[:,i]
    b = testdf_1.iloc[:,i+1]
    testdf_1['delta_'+testdf_1.iloc[:,i].name[3:]] = a - b
    
    delta_testdf['delta_'+testdf_1.iloc[:,i].name[3:]] = a - b
    i = i + 2


# **Encode categorical variables**

# In[99]:


# Create an one-hot encoder for map variables in the delta_df dataframe
# In order to transform categorical variables into dummy variables we used 'one-hot enconder'

ohe = OneHotEncoder()

delta_testdf['map'] = delta_testdf.map.astype('category')
categorical_cols = delta_testdf.columns[delta_testdf.dtypes=='category'].tolist()
hot_enc = pd.DataFrame(ohe.fit_transform(delta_testdf[categorical_cols]).toarray(), columns=ohe.get_feature_names(categorical_cols)).reset_index(drop=True)

delta_testdf = pd.concat([delta_testdf, hot_enc], axis = 1)
delta_testdf = delta_testdf.drop(['map'], axis=1)

delta_testdf.head(2)


# **Predict round winner for the test df**

# In[100]:


# generate predictions using the best-performing model
predictions = clf.predict(delta_testdf)


# In[101]:


predictions


# In[102]:


final_pred = pd.DataFrame(predictions, index = testdf.index)


# In[103]:


final_pred.to_csv('prediction_test.csv', index = True)


# #### [Get back](#tc)
