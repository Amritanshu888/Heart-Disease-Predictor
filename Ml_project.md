```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


```python
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
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```


```python
df = pd.read_csv("C:/Users/Amritanshu Bhardwaj/Downloads/data_cardiovascular_risk.csv")

# first glimpse at data
df.head(20)

# data shape
df.shape

# data types
df.dtypes
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>education</th>
      <th>sex</th>
      <th>is_smoking</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64</td>
      <td>2.0</td>
      <td>F</td>
      <td>YES</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>221.0</td>
      <td>148.0</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>90.0</td>
      <td>80.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>36</td>
      <td>4.0</td>
      <td>M</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>212.0</td>
      <td>168.0</td>
      <td>98.0</td>
      <td>29.77</td>
      <td>72.0</td>
      <td>75.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>46</td>
      <td>1.0</td>
      <td>F</td>
      <td>YES</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>250.0</td>
      <td>116.0</td>
      <td>71.0</td>
      <td>20.35</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>50</td>
      <td>1.0</td>
      <td>M</td>
      <td>YES</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>233.0</td>
      <td>158.0</td>
      <td>88.0</td>
      <td>28.26</td>
      <td>68.0</td>
      <td>94.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>64</td>
      <td>1.0</td>
      <td>F</td>
      <td>YES</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>241.0</td>
      <td>136.5</td>
      <td>85.0</td>
      <td>26.42</td>
      <td>70.0</td>
      <td>77.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>61</td>
      <td>3.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>272.0</td>
      <td>182.0</td>
      <td>121.0</td>
      <td>32.80</td>
      <td>85.0</td>
      <td>65.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>61</td>
      <td>1.0</td>
      <td>M</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>238.0</td>
      <td>232.0</td>
      <td>136.0</td>
      <td>24.83</td>
      <td>75.0</td>
      <td>79.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>36</td>
      <td>4.0</td>
      <td>M</td>
      <td>YES</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>295.0</td>
      <td>102.0</td>
      <td>68.0</td>
      <td>28.15</td>
      <td>60.0</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>41</td>
      <td>2.0</td>
      <td>F</td>
      <td>YES</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>220.0</td>
      <td>126.0</td>
      <td>78.0</td>
      <td>20.70</td>
      <td>86.0</td>
      <td>79.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>55</td>
      <td>2.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>326.0</td>
      <td>144.0</td>
      <td>81.0</td>
      <td>25.71</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>61</td>
      <td>1.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>185.0</td>
      <td>121.0</td>
      <td>35.22</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>53</td>
      <td>2.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>210.0</td>
      <td>138.0</td>
      <td>86.5</td>
      <td>22.49</td>
      <td>88.0</td>
      <td>87.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>43</td>
      <td>2.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>213.0</td>
      <td>96.0</td>
      <td>62.0</td>
      <td>19.38</td>
      <td>74.0</td>
      <td>80.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>44</td>
      <td>1.0</td>
      <td>M</td>
      <td>YES</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>227.0</td>
      <td>146.5</td>
      <td>97.0</td>
      <td>26.92</td>
      <td>80.0</td>
      <td>67.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>58</td>
      <td>3.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>188.0</td>
      <td>160.0</td>
      <td>120.0</td>
      <td>35.58</td>
      <td>88.0</td>
      <td>85.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>51</td>
      <td>1.0</td>
      <td>M</td>
      <td>YES</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>212.0</td>
      <td>146.0</td>
      <td>89.0</td>
      <td>24.49</td>
      <td>100.0</td>
      <td>132.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>50</td>
      <td>1.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>240.0</td>
      <td>163.0</td>
      <td>105.0</td>
      <td>31.37</td>
      <td>89.0</td>
      <td>75.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>44</td>
      <td>3.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>257.0</td>
      <td>129.0</td>
      <td>93.0</td>
      <td>27.56</td>
      <td>75.0</td>
      <td>76.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>56</td>
      <td>3.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>267.0</td>
      <td>122.5</td>
      <td>85.0</td>
      <td>24.22</td>
      <td>92.0</td>
      <td>100.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>42</td>
      <td>1.0</td>
      <td>M</td>
      <td>YES</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>232.0</td>
      <td>130.0</td>
      <td>91.0</td>
      <td>25.77</td>
      <td>72.0</td>
      <td>70.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






    (3390, 17)






    id                   int64
    age                  int64
    education          float64
    sex                 object
    is_smoking          object
    cigsPerDay         float64
    BPMeds             float64
    prevalentStroke      int64
    prevalentHyp         int64
    diabetes             int64
    totChol            float64
    sysBP              float64
    diaBP              float64
    BMI                float64
    heartRate          float64
    glucose            float64
    TenYearCHD           int64
    dtype: object




```python
duplicate_df = df[df.duplicated()]
duplicate_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>education</th>
      <th>sex</th>
      <th>is_smoking</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df.isna().sum()
null = df[df.isna().any(axis=1)]
null
```




    id                   0
    age                  0
    education           87
    sex                  0
    is_smoking           0
    cigsPerDay          22
    BPMeds              44
    prevalentStroke      0
    prevalentHyp         0
    diabetes             0
    totChol             38
    sysBP                0
    diaBP                0
    BMI                 14
    heartRate            1
    glucose            304
    TenYearCHD           0
    dtype: int64






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>education</th>
      <th>sex</th>
      <th>is_smoking</th>
      <th>cigsPerDay</th>
      <th>BPMeds</th>
      <th>prevalentStroke</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>totChol</th>
      <th>sysBP</th>
      <th>diaBP</th>
      <th>BMI</th>
      <th>heartRate</th>
      <th>glucose</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64</td>
      <td>2.0</td>
      <td>F</td>
      <td>YES</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>221.0</td>
      <td>148.0</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>90.0</td>
      <td>80.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>41</td>
      <td>2.0</td>
      <td>F</td>
      <td>YES</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>220.0</td>
      <td>126.0</td>
      <td>78.0</td>
      <td>20.70</td>
      <td>86.0</td>
      <td>79.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>55</td>
      <td>2.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>326.0</td>
      <td>144.0</td>
      <td>81.0</td>
      <td>25.71</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>61</td>
      <td>1.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>185.0</td>
      <td>121.0</td>
      <td>35.22</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>46</td>
      <td>3.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>193.0</td>
      <td>106.5</td>
      <td>70.5</td>
      <td>26.18</td>
      <td>75.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3349</th>
      <td>3349</td>
      <td>46</td>
      <td>2.0</td>
      <td>F</td>
      <td>NO</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>242.0</td>
      <td>129.0</td>
      <td>85.0</td>
      <td>27.40</td>
      <td>80.0</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3370</th>
      <td>3370</td>
      <td>46</td>
      <td>1.0</td>
      <td>F</td>
      <td>YES</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>219.0</td>
      <td>107.0</td>
      <td>69.0</td>
      <td>21.40</td>
      <td>66.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3378</th>
      <td>3378</td>
      <td>39</td>
      <td>3.0</td>
      <td>F</td>
      <td>YES</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>197.0</td>
      <td>126.5</td>
      <td>76.5</td>
      <td>19.71</td>
      <td>55.0</td>
      <td>63.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3379</th>
      <td>3379</td>
      <td>39</td>
      <td>1.0</td>
      <td>M</td>
      <td>YES</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>292.0</td>
      <td>120.0</td>
      <td>85.0</td>
      <td>31.09</td>
      <td>85.0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3388</th>
      <td>3388</td>
      <td>60</td>
      <td>1.0</td>
      <td>M</td>
      <td>NO</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>191.0</td>
      <td>167.0</td>
      <td>105.0</td>
      <td>23.01</td>
      <td>80.0</td>
      <td>85.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>463 rows × 17 columns</p>
</div>




```python
fig = plt.figure(figsize=(15, 20))
ax = fig.gca()
df.hist(ax=ax)
plt.show()
```

    C:\Users\Amritanshu Bhardwaj\AppData\Local\Temp\ipykernel_13884\1052702999.py:3: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared.
      df.hist(ax=ax)
    




    array([[<Axes: title={'center': 'id'}>, <Axes: title={'center': 'age'}>,
            <Axes: title={'center': 'education'}>,
            <Axes: title={'center': 'cigsPerDay'}>],
           [<Axes: title={'center': 'BPMeds'}>,
            <Axes: title={'center': 'prevalentStroke'}>,
            <Axes: title={'center': 'prevalentHyp'}>,
            <Axes: title={'center': 'diabetes'}>],
           [<Axes: title={'center': 'totChol'}>,
            <Axes: title={'center': 'sysBP'}>,
            <Axes: title={'center': 'diaBP'}>,
            <Axes: title={'center': 'BMI'}>],
           [<Axes: title={'center': 'heartRate'}>,
            <Axes: title={'center': 'glucose'}>,
            <Axes: title={'center': 'TenYearCHD'}>, <Axes: >]], dtype=object)




    
![png](Ml_project_files/Ml_project_5_2.png)
    



```python
non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
print("Non-numeric columns:", non_numeric_cols)
df_numeric = df.drop(columns=non_numeric_cols)
df_corr = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.show()

```

    Non-numeric columns: Index(['sex', 'is_smoking'], dtype='object')
    




    <Figure size 1200x1000 with 0 Axes>






    <Axes: >




    
![png](Ml_project_files/Ml_project_6_3.png)
    



```python
df.isna().sum()
```




    id                   0
    age                  0
    education           87
    sex                  0
    is_smoking           0
    cigsPerDay          22
    BPMeds              44
    prevalentStroke      0
    prevalentHyp         0
    diabetes             0
    totChol             38
    sysBP                0
    diaBP                0
    BMI                 14
    heartRate            1
    glucose            304
    TenYearCHD           0
    dtype: int64




```python
df = df.dropna()
df.isna().sum()
df.columns
```




    id                 0
    age                0
    education          0
    sex                0
    is_smoking         0
    cigsPerDay         0
    BPMeds             0
    prevalentStroke    0
    prevalentHyp       0
    diabetes           0
    totChol            0
    sysBP              0
    diaBP              0
    BMI                0
    heartRate          0
    glucose            0
    TenYearCHD         0
    dtype: int64






    Index(['id', 'age', 'education', 'sex', 'is_smoking', 'cigsPerDay', 'BPMeds',
           'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
           'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD'],
          dtype='object')




```python
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

```

               Specs       Score
    11         sysBP  519.840881
    10       totChol  278.418281
    1            age  240.058688
    0             id  199.686999
    5     cigsPerDay  162.115268
    12         diaBP   99.495351
    8   prevalentHyp   57.413962
    9       diabetes   28.483542
    6         BPMeds   24.484602
    13           BMI   11.161921
    


```python
featureScores = featureScores.sort_values(by='Score', ascending=False)
featureScores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Specs</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>sysBP</td>
      <td>519.840881</td>
    </tr>
    <tr>
      <th>10</th>
      <td>totChol</td>
      <td>278.418281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>240.058688</td>
    </tr>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>199.686999</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cigsPerDay</td>
      <td>162.115268</td>
    </tr>
    <tr>
      <th>12</th>
      <td>diaBP</td>
      <td>99.495351</td>
    </tr>
    <tr>
      <th>8</th>
      <td>prevalentHyp</td>
      <td>57.413962</td>
    </tr>
    <tr>
      <th>9</th>
      <td>diabetes</td>
      <td>28.483542</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BPMeds</td>
      <td>24.484602</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BMI</td>
      <td>11.161921</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sex</td>
      <td>10.861065</td>
    </tr>
    <tr>
      <th>7</th>
      <td>prevalentStroke</td>
      <td>7.870084</td>
    </tr>
    <tr>
      <th>2</th>
      <td>education</td>
      <td>6.005817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>is_smoking</td>
      <td>1.645271</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20, 5))
sns.barplot(x='Specs', y='Score', data=featureScores, palette="GnBu_d")
plt.box(False)
plt.title('Feature importance', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```




    <Figure size 2000x500 with 0 Axes>






    <Axes: xlabel='Specs', ylabel='Score'>






    Text(0.5, 1.0, 'Feature importance')






    Text(0.5, 0, 'Features')






    Text(0, 0.5, 'Importance')






    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]),
     [Text(0, 0, 'sysBP'),
      Text(1, 0, 'totChol'),
      Text(2, 0, 'age'),
      Text(3, 0, 'id'),
      Text(4, 0, 'cigsPerDay'),
      Text(5, 0, 'diaBP'),
      Text(6, 0, 'prevalentHyp'),
      Text(7, 0, 'diabetes'),
      Text(8, 0, 'BPMeds'),
      Text(9, 0, 'BMI'),
      Text(10, 0, 'sex'),
      Text(11, 0, 'prevalentStroke'),
      Text(12, 0, 'education'),
      Text(13, 0, 'is_smoking')])






    (array([  0., 100., 200., 300., 400., 500., 600.]),
     [Text(0, 0.0, '0'),
      Text(0, 100.0, '100'),
      Text(0, 200.0, '200'),
      Text(0, 300.0, '300'),
      Text(0, 400.0, '400'),
      Text(0, 500.0, '500'),
      Text(0, 600.0, '600')])




    
![png](Ml_project_files/Ml_project_11_7.png)
    



```python

```


```python
# selecting the 10 most impactful features for the target variable
features_list = featureScores["Specs"].tolist()[:10]
features_list
```




    ['sysBP',
     'totChol',
     'age',
     'id',
     'cigsPerDay',
     'diaBP',
     'prevalentHyp',
     'diabetes',
     'BPMeds',
     'BMI']




```python
# Create new dataframe with selected features

df = df[['sysBP','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','TenYearCHD']]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sysBP</th>
      <th>age</th>
      <th>totChol</th>
      <th>cigsPerDay</th>
      <th>diaBP</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>BPMeds</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>168.0</td>
      <td>36</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>98.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>116.0</td>
      <td>46</td>
      <td>250.0</td>
      <td>10.0</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>158.0</td>
      <td>50</td>
      <td>233.0</td>
      <td>20.0</td>
      <td>88.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>136.5</td>
      <td>64</td>
      <td>241.0</td>
      <td>30.0</td>
      <td>85.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>182.0</td>
      <td>61</td>
      <td>272.0</td>
      <td>0.0</td>
      <td>121.0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_corr = df.corr()
sns.heatmap(df_corr)
```




    <Axes: >




    
![png](Ml_project_files/Ml_project_15_1.png)
    



```python
# Checking for outliers
df.describe()
sns.pairplot(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sysBP</th>
      <th>age</th>
      <th>totChol</th>
      <th>cigsPerDay</th>
      <th>diaBP</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>BPMeds</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
      <td>2927.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>132.626409</td>
      <td>49.507345</td>
      <td>237.129142</td>
      <td>9.112743</td>
      <td>82.906218</td>
      <td>0.314315</td>
      <td>0.026990</td>
      <td>0.030065</td>
      <td>0.151691</td>
    </tr>
    <tr>
      <th>std</th>
      <td>22.326197</td>
      <td>8.597191</td>
      <td>44.613282</td>
      <td>11.882784</td>
      <td>12.078873</td>
      <td>0.464322</td>
      <td>0.162082</td>
      <td>0.170795</td>
      <td>0.358783</td>
    </tr>
    <tr>
      <th>min</th>
      <td>83.500000</td>
      <td>32.000000</td>
      <td>113.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.000000</td>
      <td>42.000000</td>
      <td>206.000000</td>
      <td>0.000000</td>
      <td>74.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>128.500000</td>
      <td>49.000000</td>
      <td>234.000000</td>
      <td>0.000000</td>
      <td>82.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>144.000000</td>
      <td>56.000000</td>
      <td>264.000000</td>
      <td>20.000000</td>
      <td>90.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>295.000000</td>
      <td>70.000000</td>
      <td>600.000000</td>
      <td>70.000000</td>
      <td>142.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <seaborn.axisgrid.PairGrid at 0x177b723ae10>




    
![png](Ml_project_files/Ml_project_16_3.png)
    



```python
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
```




    <Figure size 2000x1500 with 0 Axes>






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='sysBP', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of sysBP')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='totChol', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of totChol')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='age', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of age')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='cigsPerDay', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of cigsPerDay')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='diaBP', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of diaBP')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='prevalentHyp', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of prevalentHyp')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='diabetes', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of diabetes')






    <Axes: >



    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <Axes: xlabel='BPMeds', ylabel='Count'>






    Text(0.5, 1.0, 'Distribution of BPMeds')




    
![png](Ml_project_files/Ml_project_17_33.png)
    



```python
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_list, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(data=df, x='TenYearCHD', y=feature)
    plt.title(f'Box plot of {feature} by TenYearCHD')
plt.tight_layout()
plt.show()
```




    <Figure size 2000x1500 with 0 Axes>






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='sysBP'>






    Text(0.5, 1.0, 'Box plot of sysBP by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='totChol'>






    Text(0.5, 1.0, 'Box plot of totChol by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='age'>






    Text(0.5, 1.0, 'Box plot of age by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='cigsPerDay'>






    Text(0.5, 1.0, 'Box plot of cigsPerDay by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='diaBP'>






    Text(0.5, 1.0, 'Box plot of diaBP by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='prevalentHyp'>






    Text(0.5, 1.0, 'Box plot of prevalentHyp by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='diabetes'>






    Text(0.5, 1.0, 'Box plot of diabetes by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='BPMeds'>






    Text(0.5, 1.0, 'Box plot of BPMeds by TenYearCHD')




    
![png](Ml_project_files/Ml_project_18_25.png)
    



```python
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features_list, 1):
    plt.subplot(4, 3, i)
    sns.violinplot(data=df, x='TenYearCHD', y=feature)
    plt.title(f'Violin plot of {feature} by TenYearCHD')
plt.tight_layout()
plt.show()
```




    <Figure size 2000x1500 with 0 Axes>






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='sysBP'>






    Text(0.5, 1.0, 'Violin plot of sysBP by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='totChol'>






    Text(0.5, 1.0, 'Violin plot of totChol by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='age'>






    Text(0.5, 1.0, 'Violin plot of age by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='cigsPerDay'>






    Text(0.5, 1.0, 'Violin plot of cigsPerDay by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='diaBP'>






    Text(0.5, 1.0, 'Violin plot of diaBP by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='prevalentHyp'>






    Text(0.5, 1.0, 'Violin plot of prevalentHyp by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='diabetes'>






    Text(0.5, 1.0, 'Violin plot of diabetes by TenYearCHD')






    <Axes: >






    <Axes: xlabel='TenYearCHD', ylabel='BPMeds'>






    Text(0.5, 1.0, 'Violin plot of BPMeds by TenYearCHD')




    
![png](Ml_project_files/Ml_project_19_25.png)
    



```python
top_5_features = features_list[:5]
sns.pairplot(df[top_5_features + ['TenYearCHD']], diag_kind='kde', hue='TenYearCHD')
plt.show()
```

    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\Amritanshu Bhardwaj\anaconda3\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    




    <seaborn.axisgrid.PairGrid at 0x18895d55fd0>




    
![png](Ml_project_files/Ml_project_20_2.png)
    



```python
# Zooming into cholesterin outliers

sns.boxplot(data=df,x='totChol')
outliers = df[(df['totChol'] > 500)] 
outliers
```




    <Axes: xlabel='totChol'>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sysBP</th>
      <th>age</th>
      <th>totChol</th>
      <th>cigsPerDay</th>
      <th>diaBP</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>BPMeds</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>423</th>
      <td>159.5</td>
      <td>52</td>
      <td>600.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](Ml_project_files/Ml_project_21_2.png)
    



```python
df = df.drop(df[df['totChol'] > 599].index)
sns.boxplot(data=df,x='totChol')
```




    <Axes: xlabel='totChol'>




    
![png](Ml_project_files/Ml_project_22_1.png)
    



```python
df_clean = df
```


```python
#Feature scaling
scaler = MinMaxScaler(feature_range=(0,1)) 

#assign scaler to column:
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)
```


```python
df_scaled.describe()
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sysBP</th>
      <th>age</th>
      <th>totChol</th>
      <th>cigsPerDay</th>
      <th>diaBP</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>BPMeds</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.232233</td>
      <td>0.460697</td>
      <td>0.353291</td>
      <td>0.130227</td>
      <td>0.369338</td>
      <td>0.314081</td>
      <td>0.026658</td>
      <td>0.030075</td>
      <td>0.151401</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.105553</td>
      <td>0.226277</td>
      <td>0.125679</td>
      <td>0.169766</td>
      <td>0.127822</td>
      <td>0.464228</td>
      <td>0.161108</td>
      <td>0.170823</td>
      <td>0.358501</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.158392</td>
      <td>0.263158</td>
      <td>0.264957</td>
      <td>0.000000</td>
      <td>0.280423</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.212766</td>
      <td>0.447368</td>
      <td>0.344729</td>
      <td>0.000000</td>
      <td>0.359788</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.286052</td>
      <td>0.631579</td>
      <td>0.430199</td>
      <td>0.285714</td>
      <td>0.444444</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sysBP</th>
      <th>age</th>
      <th>totChol</th>
      <th>cigsPerDay</th>
      <th>diaBP</th>
      <th>prevalentHyp</th>
      <th>diabetes</th>
      <th>BPMeds</th>
      <th>TenYearCHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
      <td>2926.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>132.617225</td>
      <td>49.506494</td>
      <td>237.005126</td>
      <td>9.115858</td>
      <td>82.902427</td>
      <td>0.314081</td>
      <td>0.026658</td>
      <td>0.030075</td>
      <td>0.151401</td>
    </tr>
    <tr>
      <th>std</th>
      <td>22.324482</td>
      <td>8.598536</td>
      <td>44.113408</td>
      <td>11.883620</td>
      <td>12.079196</td>
      <td>0.464228</td>
      <td>0.161108</td>
      <td>0.170823</td>
      <td>0.358501</td>
    </tr>
    <tr>
      <th>min</th>
      <td>83.500000</td>
      <td>32.000000</td>
      <td>113.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>117.000000</td>
      <td>42.000000</td>
      <td>206.000000</td>
      <td>0.000000</td>
      <td>74.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>128.500000</td>
      <td>49.000000</td>
      <td>234.000000</td>
      <td>0.000000</td>
      <td>82.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>144.000000</td>
      <td>56.000000</td>
      <td>264.000000</td>
      <td>20.000000</td>
      <td>90.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>295.000000</td>
      <td>70.000000</td>
      <td>464.000000</td>
      <td>70.000000</td>
      <td>142.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Train test split
# clarify what is y and what is x label
y = df_scaled['TenYearCHD']
X = df_scaled.drop(['TenYearCHD'], axis = 1)

# divide train test: 80 % - 20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)
```


```python
len(X_train)
len(X_test)
```




    2340






    586




```python
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
```

    Class 0: 2483
    Class 1: 443
    Proportion: 5.6 : 1
    




    <Axes: ylabel='count'>






    Text(0.5, 0, 'Heart Disease No/Yes')






    Text(0, 0.5, 'Patient Count')






    Text(0.5, 1.0, 'Count Outcome Heart Disease\n')




    
![png](Ml_project_files/Ml_project_28_5.png)
    



```python
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
```




    TenYearCHD
    0.0    611
    1.0    443
    Name: count, dtype: int64






    <Axes: xlabel='TenYearCHD', ylabel='count'>






    Text(0.5, 0, 'Heart Disease No/Yes')






    Text(0, 0.5, 'Patient Count')






    Text(0.5, 1.0, 'Count Outcome Heart Disease after Resampling\n')




    
![png](Ml_project_files/Ml_project_29_5.png)
    



```python
y_train = normalized_df['TenYearCHD']
X_train = normalized_df.drop('TenYearCHD', axis=1)

from sklearn.pipeline import Pipeline

classifiers = [LogisticRegression(),SVC(),DecisionTreeClassifier(),KNeighborsClassifier(2)]

for classifier in classifiers:
    pipe = Pipeline(steps=[('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print("The accuracy score of {0} is: {1:.2f}%".format(classifier,(pipe.score(X_test, y_test)*100)))
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, LogisticRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, LogisticRegression())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div>



    The accuracy score of LogisticRegression() is: 74.40%
    




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, SVC())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, SVC())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC()</pre></div></div></div></div></div></div></div>



    The accuracy score of SVC() is: 74.23%
    




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, DecisionTreeClassifier())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, DecisionTreeClassifier())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div></div></div>



    The accuracy score of DecisionTreeClassifier() is: 77.65%
    




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, KNeighborsClassifier(n_neighbors=2))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;classifier&#x27;, KNeighborsClassifier(n_neighbors=2))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=2)</pre></div></div></div></div></div></div></div>



    The accuracy score of KNeighborsClassifier(n_neighbors=2) is: 81.57%
    


```python
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
```

    The accuracy score for LogReg is: 74.4%
    The f1 score for LogReg is: 41.9%
    The precision score for LogReg is: 34.0%
    The recall score for LogReg is: 54.50000000000001%
    


```python
# plotting confusion matrix LogReg

cnf_matrix_log = confusion_matrix(y_test, normalized_df_reg_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Logistic Regression\n', y=1.1)
```




    <Axes: >






    Text(0.5, 1.1, 'Confusion matrix Logistic Regression\n')




    
![png](Ml_project_files/Ml_project_32_2.png)
    



```python
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
```




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" checked><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC()</pre></div></div></div></div></div>



    The accuracy score for SVM is: 74.2%
    The f1 score for SVM is: 41.699999999999996%
    The precision score for SVM is: 33.800000000000004%
    The recall score for SVM is: 54.50000000000001%
    


```python
# plotting confusion matrix SVM

cnf_matrix_svm = confusion_matrix(y_test, normalized_df_svm_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix SVM\n', y=1.1)
```




    <Axes: >






    Text(0.5, 1.1, 'Confusion matrix SVM\n')




    
![png](Ml_project_files/Ml_project_34_2.png)
    



```python
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
```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" checked><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>



    The accuracy score for DTC is: 78.2%
    The f1 score for DTC is: 60.699999999999996%
    The precision score for DTC is: 43.6%
    The recall score for DTC is: 100.0%
    


```python
# plotting confusion matrix Decision Tree

cnf_matrix_dtc = confusion_matrix(y_test, normalized_df_dtc_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_dtc), annot=True,cmap="Reds" , fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix Decision Tree\n', y=1.1)
```




    <Axes: >






    Text(0.5, 1.1, 'Confusion matrix Decision Tree\n')




    
![png](Ml_project_files/Ml_project_36_2.png)
    



```python
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


```




<style>#sk-container-id-10 {color: black;background-color: white;}#sk-container-id-10 pre{padding: 0;}#sk-container-id-10 div.sk-toggleable {background-color: white;}#sk-container-id-10 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-10 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-10 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-10 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-10 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-10 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-10 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-10 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-10 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-10 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-10 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-10 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-10 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-10 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-10 div.sk-item {position: relative;z-index: 1;}#sk-container-id-10 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-10 div.sk-item::before, #sk-container-id-10 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-10 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-10 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-10 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-10 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-10 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-10 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-10 div.sk-label-container {text-align: center;}#sk-container-id-10 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-10 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-10" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" checked><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=2)</pre></div></div></div></div></div>



    The accuracy score for KNN is: 82.1%
    The f1 score for KNN is: 7.1%
    The precision score for KNN is: 28.599999999999998%
    The recall score for KNN is: 4.0%
    


```python
#Result: The KNN model has the highest accuracy score
```


```python
# Check overfit of the KNN model
# accuracy test and train
acc_test = knn.score(X_test, y_test)
print("The accuracy score of the test data is: ",acc_test*100,"%")
acc_train = knn.score(X_train, y_train)
print("The accuracy score of the training data is: ",round(acc_train*100,2),"%")
```

    The accuracy score of the test data is:  82.08191126279864 %
    The accuracy score of the training data is:  88.33 %
    


```python
# Perform cross validation
'''Cross Validation is used to assess the predictive performance of the models and and to judge 
how they perform outside the sample to a new data set'''

cv_results = cross_val_score(knn, X, y, cv=5) 

print ("Cross-validated scores:", cv_results)
print("The Accuracy of Model with Cross Validation is: {0:.2f}%".format(cv_results.mean() * 100))
```




    'Cross Validation is used to assess the predictive performance of the models and and to judge \nhow they perform outside the sample to a new data set'



    Cross-validated scores: [0.83788396 0.83247863 0.84444444 0.83418803 0.84102564]
    The Accuracy of Model with Cross Validation is: 83.80%
    


```python
cnf_matrix_knn = confusion_matrix(y_test, normalized_df_knn_pred)

ax= plt.subplot()
sns.heatmap(pd.DataFrame(cnf_matrix_knn), annot=True,cmap="Reds" , fmt='g')

ax.set_xlabel('Predicted ');ax.set_ylabel('True'); 
```


    
![png](Ml_project_files/Ml_project_41_0.png)
    



```python
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
```




    "the AUC ROC Curve is a measure of performance based on plotting the true positive and false positive rate \nand calculating the area under that curve.The closer the score to 1 the better the algorithm's ability to \ndistinguish between the two outcome classes."






    [<matplotlib.lines.Line2D at 0x1889c2e7e50>]






    <matplotlib.legend.Legend at 0x1889ea77350>






    Text(0.5, 1.0, 'ROC CURVE KNN')




    
![png](Ml_project_files/Ml_project_42_4.png)
    


    The score for the AUC ROC Curve is: 51.0%
    


```python
fpr, tpr, _ = roc_curve(y_test, normalized_df_dtc_pred)
auc = roc_auc_score(y_test, normalized_df_dtc_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.box(False)
plt.title ('ROC CURVE Decision Tree')
plt.show()
```




    [<matplotlib.lines.Line2D at 0x18896ffc6d0>]






    <matplotlib.legend.Legend at 0x18897083d90>






    Text(0.5, 1.0, 'ROC CURVE Decision Tree')




    
![png](Ml_project_files/Ml_project_43_3.png)
    



```python
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

```




    <Axes: xlabel='totChol'>



    Empty DataFrame
    Columns: [sysBP, age, totChol, cigsPerDay, diaBP, prevalentHyp, diabetes, BPMeds, TenYearCHD]
    Index: []
    




    <Axes: xlabel='totChol'>



                 sysBP          age      totChol   cigsPerDay        diaBP  \
    count  1054.000000  1054.000000  1054.000000  1054.000000  1054.000000   
    mean      0.249784     0.501898     0.353202     0.150364     0.413527   
    std       0.117659     0.248963     0.130887     0.198448     0.149479   
    min       0.000000     0.000000     0.000000     0.000000     0.000000   
    25%       0.167849     0.277778     0.263768     0.000000     0.312500   
    50%       0.225768     0.500000     0.344928     0.000000     0.397727   
    75%       0.309693     0.722222     0.428986     0.333333     0.500000   
    max       1.000000     1.000000     1.000000     1.000000     1.000000   
    
           prevalentHyp     diabetes       BPMeds   TenYearCHD  
    count   1054.000000  1054.000000  1054.000000  1054.000000  
    mean       0.380455     0.037002     0.038899     0.420304  
    std        0.485729     0.188856     0.193447     0.493842  
    min        0.000000     0.000000     0.000000     0.000000  
    25%        0.000000     0.000000     0.000000     0.000000  
    50%        0.000000     0.000000     0.000000     0.000000  
    75%        1.000000     0.000000     0.000000     1.000000  
    max        1.000000     1.000000     1.000000     1.000000  
                 sysBP          age      totChol   cigsPerDay        diaBP  \
    count  2926.000000  2926.000000  2926.000000  2926.000000  2926.000000   
    mean    132.617225    49.506494   237.005126     9.115858    82.902427   
    std      22.324482     8.598536    44.113408    11.883620    12.079196   
    min      83.500000    32.000000   113.000000     0.000000    48.000000   
    25%     117.000000    42.000000   206.000000     0.000000    74.500000   
    50%     128.500000    49.000000   234.000000     0.000000    82.000000   
    75%     144.000000    56.000000   264.000000    20.000000    90.000000   
    max     295.000000    70.000000   464.000000    70.000000   142.500000   
    
           prevalentHyp     diabetes       BPMeds   TenYearCHD  
    count   2926.000000  2926.000000  2926.000000  2926.000000  
    mean       0.314081     0.026658     0.030075     0.151401  
    std        0.464228     0.161108     0.170823     0.358501  
    min        0.000000     0.000000     0.000000     0.000000  
    25%        0.000000     0.000000     0.000000     0.000000  
    50%        0.000000     0.000000     0.000000     0.000000  
    75%        1.000000     0.000000     0.000000     0.000000  
    max        1.000000     1.000000     1.000000     1.000000  
    




<style>#sk-container-id-11 {color: black;background-color: white;}#sk-container-id-11 pre{padding: 0;}#sk-container-id-11 div.sk-toggleable {background-color: white;}#sk-container-id-11 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-11 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-11 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-11 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-11 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-11 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-11 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-11 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-11 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-11 div.sk-item {position: relative;z-index: 1;}#sk-container-id-11 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-11 div.sk-item::before, #sk-container-id-11 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-11 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-11 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-11 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-11 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-11 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-11 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-11 div.sk-label-container {text-align: center;}#sk-container-id-11 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-11 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-11" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" checked><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier(n_neighbors=2)</pre></div></div></div></div></div>



    The accuracy score for KNN is: 59.199999999999996%
    The F1 score for KNN is: 33.800000000000004%
    The precision score for KNN is: 53.7%
    The recall score for KNN is: 24.7%
    


    
![png](Ml_project_files/Ml_project_44_6.png)
    



```python
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



```

    Input Patient Information:
    

    Patient's systolic blood pressure: >>>  129
    Patient's age: >>>  34
    Patient's cholesterol level: >>>  89
    Patient's smoked cigarettes per day: >>>  2
    Patient's diastolic blood pressure: >>>  89
    Was Patient hypertensive? Yes=1, No=0 >>>  1
    Did Patient have diabetes? Yes=1, No=0 >>>  0
    Has Patient been on Blood Pressure Medication? Yes=1, No=0 >>>  0
    

    
    Result:
    The patient will not develop a Heart Disease.
    

### 
