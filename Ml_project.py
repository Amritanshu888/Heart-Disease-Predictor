#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, recall_score, 
                             precision_score, roc_curve, roc_auc_score)
from sklearn.feature_selection import SelectKBest, chi2
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("C:/Users/Amritanshu Bhardwaj/Downloads/data_cardiovascular_risk.csv")

# first glimpse at data
df.head(20)

# data shape
df.shape

# data types
df.dtypes


# In[4]:


duplicate_df = df[df.duplicated()]
duplicate_df


# In[5]:


df.isna().sum()
null = df[df.isna().any(axis=1)]
null


# In[6]:


fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
df.hist(ax=ax)
plt.show()


# In[7]:


non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
print("Non-numeric columns:", non_numeric_cols)
df_numeric = df.drop(columns=non_numeric_cols)
df_corr = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.show()


# In[8]:


df.isna().sum()


# In[9]:


df = df.dropna()
df.isna().sum()
df.columns


# In[10]:


# Identify the features with the most importance for the outcome variable Heart Disease

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Convert categorical variables to numeric using LabelEncoder
df_encoded = df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)

# Separate independent & dependent variables
X = df_encoded.iloc[:, 0:14]  # independent columns
y = df_encoded.iloc[:, -1]    # target column i.e. price range

# Apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concatenate two dataframes for better visualization 
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))


# In[11]:


featureScores = featureScores.sort_values(by='Score', ascending=False)
featureScores


# In[12]:


plt.figure(figsize=(20, 5))
sns.barplot(x='Specs', y='Score', data=featureScores, palette="GnBu_d")
plt.box(False)
plt.title('Feature importance', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[ ]:





# In[13]:


# selecting the 10 most impactful features for the target variable
features_list = featureScores["Specs"].tolist()[:10]
features_list


# In[14]:


# Create new dataframe with selected features

df = df[['sysBP','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','TenYearCHD']]
df.head()


# In[15]:


df_corr = df.corr()
sns.heatmap(df_corr)


# In[19]:


# Checking for outliers
df.describe()
sns.pairplot(df)


# In[17]:


features_list = [feature for feature in features_list if feature in df.columns]

# Additional visualizations
# Distribution plots for selected features
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_list, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[18]:


plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_list, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df, x='TenYearCHD', y=feature)
    plt.title(f'Box plot of {feature} by TenYearCHD')
plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_list, 1):
    plt.subplot(4, 3, i)
    sns.violinplot(data=df, x='TenYearCHD', y=feature)
    plt.title(f'Violin plot of {feature} by TenYearCHD')
plt.tight_layout()
plt.show()


# In[20]:


top_5_features = features_list[:5]
sns.pairplot(df[top_5_features + ['TenYearCHD']], diag_kind='kde', hue='TenYearCHD')
plt.show()


# In[21]:


# Zooming into cholesterin outliers

sns.boxplot(data=df,x='totChol')
outliers = df[(df['totChol'] > 500)] 
outliers


# In[22]:


df = df.drop(df[df['totChol'] > 599].index)
sns.boxplot(data=df,x='totChol')


# In[23]:


df_clean = df


# In[24]:


#Feature scaling
scaler = MinMaxScaler(feature_range=(0,1)) 

#assign scaler to column:
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)


# In[25]:


df_scaled.describe()
df.describe()


# In[26]:


#Train test split
# clarify what is y and what is x label
y = df_scaled['TenYearCHD']
X = df_scaled.drop(['TenYearCHD'], axis = 1)

# divide train test: 80 % - 20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)


# In[27]:


len(X_train)
len(X_test)


# In[28]:


#Resampling imbalanced Dataset 
# Checking balance of outcome variable
target_count = df_scaled.TenYearCHD.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

sns.countplot(df_scaled.TenYearCHD, palette="OrRd")
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease\n')
plt.savefig('Balance Heart Disease.png')
plt.show()


# In[29]:


# Shuffle df
shuffled_df = df_scaled.sample(frac=1,random_state=4)

# Put all the fraud class in a separate dataset.
CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 1]

#Randomly select 492 observations from the non-fraud (majority class)
non_CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 0].sample(n=611,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([CHD_df, non_CHD_df])

# check new class counts
normalized_df.TenYearCHD.value_counts()

# plot new count
sns.countplot(x='TenYearCHD',data=normalized_df)
plt.box(False)
plt.xlabel('Heart Disease No/Yes',fontsize=11)
plt.ylabel('Patient Count',fontsize=11)
plt.title('Count Outcome Heart Disease after Resampling\n')
#plt.savefig('Balance Heart Disease.png')
plt.show()


# In[30]:


y_train = normalized_df['TenYearCHD']
X_train = normalized_df.drop('TenYearCHD', axis=1)

from sklearn.pipeline import Pipeline

classifiers = [LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(2)]

for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print("The accuracy score of {0} is: {1:.2f}%".format(classifier,(pipe.score(X_test, y_test)*100)))


# In[31]:


# logistic regression again with the balanced dataset

normalized_df_reg = LogisticRegression().fit(X_train, y_train)

normalized_df_reg_pred = normalized_df_reg.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_reg_pred)
print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_reg_pred)
print(f"The f1 score for LogReg is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_reg_pred)
print(f"The precision score for LogReg is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_reg_pred)
print(f"The recall score for LogReg is: {round(recall,3)*100}%")


# In[32]:


# plotting confusion matrix LogReg

cnf_matrix_log = confusion_matrix(y_test, normalized_df_reg_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Logistic Regression\n', y=1.1)


# In[33]:


# Support Vector Machine

#initialize model
svm = SVC()

#fit model
svm.fit(X_train, y_train)

normalized_df_svm_pred = svm.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_svm_pred)
print(f"The accuracy score for SVM is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_svm_pred)
print(f"The f1 score for SVM is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_svm_pred)
print(f"The precision score for SVM is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_svm_pred)
print(f"The recall score for SVM is: {round(recall,3)*100}%")


# In[34]:


# plotting confusion matrix SVM

cnf_matrix_svm = confusion_matrix(y_test, normalized_df_svm_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix SVM\n', y=1.1)


# In[35]:


# Decision Tree

#initialize model
dtc_up = DecisionTreeClassifier()

# fit model
dtc_up.fit(X_train, y_train)

normalized_df_dtc_pred = dtc_up.predict(X_test)

# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_dtc_pred)
print(f"The accuracy score for DTC is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_dtc_pred)
print(f"The f1 score for DTC is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_dtc_pred)
print(f"The precision score for DTC is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_dtc_pred)
print(f"The recall score for DTC is: {round(recall,3)*100}%")


# In[36]:


# plotting confusion matrix Decision Tree

cnf_matrix_dtc = confusion_matrix(y_test, normalized_df_dtc_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_dtc), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Decision Tree\n', y=1.1)


# In[62]:


# KNN Model
knn = KNeighborsClassifier(n_neighbors = 2)

#fit model
knn.fit(X_train, y_train)

# prediction = knn.predict(x_test)
normalized_df_knn_pred = knn.predict(X_test)


# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total
acc = accuracy_score(y_test, normalized_df_knn_pred)
print(f"The accuracy score for KNN is: {round(acc,3)*100}%")

# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
f1 = f1_score(y_test, normalized_df_knn_pred)
print(f"The f1 score for KNN is: {round(f1,3)*100}%")

# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes
precision = precision_score(y_test, normalized_df_knn_pred)
print(f"The precision score for KNN is: {round(precision,3)*100}%")

# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes
recall = recall_score(y_test, normalized_df_knn_pred)
print(f"The recall score for KNN is: {round(recall,3)*100}%")



# In[63]:


#Result: The KNN model has the highest accuracy score


# In[64]:


# Check overfit of the KNN model
# accuracy test and train
acc_test = knn.score(X_test, y_test)
print("The accuracy score of the test data is: ",acc_test*100,"%")
acc_train = knn.score(X_train, y_train)
print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")


# In[65]:


# Perform cross validation
'''Cross Validation is used to assess the predictive performance of the models and and to judge 
how they perform outside the sample to a new data set'''

cv_results = cross_val_score(knn, X, y, cv=5) 

print ("Cross-validated scores:", cv_results)
print("The Accuracy of Model with Cross Validation is: {0:.2f}%".format(cv_results.mean() * 100))


# In[66]:


cnf_matrix_knn = confusion_matrix(y_test, normalized_df_knn_pred)

ax= plt.subplot()
sns.heatmap(pd.DataFrame(cnf_matrix_knn), annot=True,cmap="Reds" , fmt='g')

ax.set_xlabel('Predicted ');ax.set_ylabel('True'); 


# In[67]:


# AU ROC CURVE KNN
'''the AUC ROC Curve is a measure of performance based on plotting the true positive and false positive rate 
and calculating the area under that curve.The closer the score to 1 the better the algorithm's ability to 
distinguish between the two outcome classes.'''

fpr, tpr, _ = roc_curve(y_test, normalized_df_knn_pred)
auc = roc_auc_score(y_test, normalized_df_knn_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.box(False)
plt.title ('ROC CURVE KNN')
plt.show()

print(f"The score for the AUC ROC Curve is: {round(auc,3)*100}%")


# In[68]:


fpr, tpr, _ = roc_curve(y_test, normalized_df_dtc_pred)
auc = roc_auc_score(y_test, normalized_df_dtc_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.box(False)
plt.title ('ROC CURVE Decision Tree')
plt.show()


# In[72]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns

# Assume df is your dataframe

# Plot initial boxplot for totChol
sns.boxplot(data=normalized_df, x='totChol')

# Identify outliers
outliers = normalized_df[(normalized_df['totChol'] > 500)]
print(outliers)

# Remove outliers
normalized_df = normalized_df.drop(normalized_df[normalized_df['totChol'] > 599].index)

# Plot boxplot again after removing outliers
sns.boxplot(data=normalized_df, x='totChol')

# Save clean data
df_clean = normalized_df

# Feature scaling
scaler = MinMaxScaler(feature_range=(0,1))

# Assign scaler to columns
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
print(df_scaled.describe())

# Print original dataframe description
print(df.describe())

# Define features and target
y = df_scaled['TenYearCHD']
X = df_scaled.drop(['TenYearCHD'], axis=1)

# Split data into training and testing sets (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Predict using the test set
normalized_df_knn_pred = knn.predict(X_test)

# Calculate and print accuracy
acc = accuracy_score(y_test, normalized_df_knn_pred)
print(f"The accuracy score for KNN is: {round(acc, 3) * 100}%")

# Calculate and print F1 score
f1 = f1_score(y_test, normalized_df_knn_pred)
print(f"The F1 score for KNN is: {round(f1, 3) * 100}%")

# Calculate and print precision
precision = precision_score(y_test, normalized_df_knn_pred)
print(f"The precision score for KNN is: {round(precision, 3) * 100}%")

# Calculate and print recall
recall = recall_score(y_test, normalized_df_knn_pred)
print(f"The recall score for KNN is: {round(recall, 3) * 100}%")


# In[75]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def start_questionnaire():
    # Define the exact feature names as used during model training
    parameters = ['sysBP', 'age', 'totChol', 'cigsPerDay', 'diaBP', 'prevalentHyp', 'diabetes', 'BPMeds']
    
    print('Input Patient Information:')
    
    # Collect the necessary inputs
    sysBP = input("Patient's systolic blood pressure: >>> ")
    age = input("Patient's age: >>> ")
    totChol = input("Patient's cholesterol level: >>> ")
    cigsPerDay = input("Patient's smoked cigarettes per day: >>> ")
    diaBP = input("Patient's diastolic blood pressure: >>> ")
    prevalentHyp = input("Was Patient hypertensive? Yes=1, No=0 >>> ")
    diabetes = input("Did Patient have diabetes? Yes=1, No=0 >>> ")
    BPMeds = input("Has Patient been on Blood Pressure Medication? Yes=1, No=0 >>> ")
    
    # Prepare the data
    my_predictors = [
        float(sysBP), float(age), float(totChol), int(cigsPerDay), 
        float(diaBP), int(prevalentHyp), int(diabetes), int(BPMeds)
    ]
    
    my_data = dict(zip(parameters, my_predictors))
    my_df = pd.DataFrame(my_data, index=[0])
    
    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    my_df_scaled = pd.DataFrame(scaler.fit_transform(my_df), columns=my_df.columns)
    
    # Ensure knn model and scaler are already defined and trained
    try:
        my_y_pred = knn.predict(my_df_scaled)
        print('\nResult:')
        if my_y_pred[0] == 1:
            print("The patient will develop a Heart Disease.")
        else:
            print("The patient will not develop a Heart Disease.")
    except NameError:
        print("Error: KNN model (knn) is not defined. Please ensure the model is trained and available.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Uncomment the line below to test the function if the model `knn` is defined
start_questionnaire()




# ### 
