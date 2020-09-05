---
title: "Customer Churn Prediction"
date: 2020-06-28
tags: [machine learning, data science, eda, streamlit]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Data Science, EDA"
mathjax: "true"
---


Why Customer retention is important? [source](https://www.dcrstrategies.com/customer-incentives/5-reasons-customer-retention-business/)

    1. Save Money On Marketing
    2. Repeat Purchases From Repeat Customers Means Repeat Profit
    3. Free Word-Of-Mouth Advertising
    4. Retained Customers Will Provide Valuable Feedback
    5. Previous Customers Will Pay Premium Prices. 

Why and when will a customer leave his/her bank could be a challenging question to answer.

Here, we have a data from kaggle where all the historical information about a customer and whether he/she left the bank or not is available.

Our goal is to use the power of data science to help the bank identify those who are likely to leave the bank in future.


## Read Data
```python
# source - https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers
bank_data = pd.read_csv('Churn_Modelling.csv')
bank_data.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


```python
# Dimensions
print("Number of Rows: {} \nNumber of Columns: {}".format(bank_data.shape[0],bank_data.shape[1]))
```
Number of Rows: 10000 

Number of Columns: 14
  
```python
# data types, missing values and number of uniques
bank_data_info = pd.concat([pd.DataFrame(bank_data.dtypes),pd.DataFrame(bank_data.isnull().sum()),pd.DataFrame(bank_data.nunique())],axis = 1)
bank_data_info.columns = ['DataType','# missing rows','# Unique values']
print(bank_data_info)
del bank_data_info

```
                    DataType  # missing rows  # Unique values
    RowNumber          int64               0            10000
    CustomerId         int64               0            10000
    Surname           object               0             2932
    CreditScore        int64               0              460
    Geography         object               0                3
    Gender            object               0                2
    Age                int64               0               70
    Tenure             int64               0               11
    Balance          float64               0             6382
    NumOfProducts      int64               0                4
    HasCrCard          int64               0                2
    IsActiveMember     int64               0                2
    EstimatedSalary  float64               0             9999
    Exited             int64               0                2
    
The data has 10000 rows and columns. Let's see the data description.

## Data Description

    1. RowNumber: Just a index number assigned to each row. Type : int64
    2. CustomerId: Id of each customer of the bank. Type : int64
    3. Surname: Surname of the customer. Type : Object
    4. CreditScore: The measure of an individual's ability to payback the borrowed amount. Higher it is the better. Type : int64
    5. Geography: Country of the customer. Type : Object. Values: [France, Germany, Spain]
    6. Gender: Customer's gender. Type : Object. Values: [Male / Female]
    7. Age: Age of the customer. Type : int64
    8. Tenure: Duration for which the loan amount is sanctioned.Assuming it to be in years Type : int64
    9. Balance: The amount of money the customer has available in his account. Type: int64
    10. NumOfProducts: How many accounts, bank account affiliated products the person has. Type: int64
    11. HasCrCard: whether the person holds a credit card or not. 1 means he/she has a credit card and 0 means he/she doesn't. Type: int64
    12. IsActiveMember: Whether the customer is actively using the account. However, the values are subjective. Type: int64
    13. EstimatedSalary: The person's approximate salary. Type: float64
    14. Exited: Whether the customer has left the bank or not. 1 means he/she left and 0 means he/she didn't. Type: int64

From the above, we will not require RowNumber, CustomerId, and Surname are related to individuals.


## Memory Handling
Memory usage in python is a key task. In case of huge datasets memory handling is not easy. It is always a good practice to reduce memory of the data.

```python
# Before Memory reduction
print("Total memory used before Memory reduction {:5.2f}Mb".format(bank_data.memory_usage().sum() / 1024**2))
```

Total memory used before Memory reduction  0.84Mb
    
```python
# After Memory reduction
bank_data = reduce_memory(bank_data)
print("Total memory used after Memory reduction {:5.2f}Mb".format(bank_data.memory_usage().sum() / 1024**2))
```

Memory usage decreased to  0.31 Mb (63.6% reduction)
Total memory used after Memory reduction  0.31Mb
    

## Exploratory Data Analysis (EDA)
The purpose of EDA is to understand how different variables are related to our target (Exited) variable.

```python
import plotly.graph_objects as go

labels = ['Exited','Continued']
values =  [bank_data.Exited[bank_data['Exited']==1].count(), bank_data.Exited[bank_data['Exited']==0].count()]
colors = ['red', 'darkorange']
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hole=.4, hoverinfo='label+value',  textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    title_text="Ratio of Customer Churned and Retained")
fig.show()
fig.write_html(fig, file='pie_chart.html', auto_open=True)

```
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/pie_chart.png" alt="Pie Chart">


### Gender 

Who is loyal to the bank? Male or Female?


### Gender,HasCrCard,IsActiveMember vs Churn
```python
f,ax=plt.subplots(3,2,figsize=(18,25))
bank_data[['Gender','Exited']].groupby(['Gender']).mean().plot.bar(ax=ax[0][0])
ax[0][1].set_title('Churn vs Gender')
sns.countplot('Gender',hue='Exited',data=bank_data,ax=ax[0][1])
ax[0][1].set_title('Gender:Churned vs Retained')
bank_data[['HasCrCard','Exited']].groupby(['HasCrCard']).mean().plot.bar(ax=ax[1][0])
ax[1][0].set_title('Churn vs HasCrCard')
sns.countplot('HasCrCard',hue='Exited',data=bank_data,ax=ax[1][1])
ax[1][1].set_title('HasCrCard: Churned vs Retained')
bank_data[['IsActiveMember','Exited']].groupby(['IsActiveMember']).mean().plot.bar(ax=ax[2][0])
ax[2][0].set_title('Churn vs IsActiveMember')
sns.countplot('IsActiveMember',hue='Exited',data=bank_data,ax=ax[2][1])
ax[2][1].set_title('IsActiveMember: Churned vs Retained')
plt.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_22_0.png" alt="Gender,HasCrCard,IsActiveMember vs Churn">

From the above graphs we can see, 

1. More male customers, but when it comes to churn rate, female customers are more likely to quit the bank. (In other words, even though there are more male customers its the females who have high churn rate compared to males).

2. Majority of customers have credit cards. 

3. The bank have a significant number of inactive customers. They ratio of inactive customers being churned out is high. Thus bank needs to take steps and make them active. 

### Geography vs Churn

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_24_1.png" alt="Geography vs Churn">

The bank have majority of its customers located in France, however the chrun rate is high in Germany followed by spain, where the bank have less number of customers. This can be due to less number of branches in Germany and Spain or poor services in those regions. 

### Age
```python
# peaks for Exited/not exited customers by their age
facet = sns.FacetGrid(bank_data, hue="Exited",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, bank_data['Age'].max()))
facet.add_legend()

# average exited customers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = bank_data[["Age", "Exited"]].groupby(['Age'],as_index=False).mean()
average_age.columns = ['Age','Mean(Exited)']
sns.barplot(x='Age', y='Mean(Exited)', data=average_age)
del average_age
```

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_26_0.png" alt="Peaks for age">

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_26_1.png" alt="average exited age">

Customer having age around 48 to 60 are churning out compared to younger ones i.e., Mean(Exited) > 0.5 from the graph. The churn rate can also be due to retirement. Bank needs to revise the market strategy by focusing on keeping older customers. 


```python
bank_data[(bank_data['Age'] == 56)]['Exited'].value_counts()
```

    1    50
    0    20
    Name: Exited, dtype: int64

Out of 70 people whose age equals 56, 50 of them churned out.

### Tenure
```python
y0 = bank_data.Tenure[bank_data.Exited == 0].values
y1 = bank_data.Tenure[bank_data.Exited == 1].values

fig = go.Figure()
fig.add_trace(go.Box(y=y0, name='Continued',
                marker_color = 'lightseagreen'))
fig.add_trace(go.Box(y=y1, name = 'Exited',
                marker_color = 'indianred'))

fig.update_layout(
    yaxis_title='Tenure'
)

fig.show()
```
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/tenure.png" alt="Tenure">

With respect to Tenure, Customers repaying loans in less years or taking more time to repay loans are churning out. Bank needs to provide benefits for customers who are repaying loans in quick time and for those who have stayed for a long time (High tenure). 


### Balance
```python
y0 = bank_data.Balance[bank_data.Exited == 0].values
y1 = bank_data.Balance[bank_data.Exited == 1].values

fig = go.Figure()
fig.add_trace(go.Box(y=y0, name='Continued',
                marker_color = 'blue'))
fig.add_trace(go.Box(y=y1, name = 'Exited',
                marker_color = 'red'))

fig.update_layout(
    yaxis_title='Balance'
)

fig.show()
```

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/balance.png" alt="Balance">

We can see that Customers having balance even after churning out.

Even though the gap is not very high yet the Customers with high account balance are churning out. This may even be due to the data imbalance. To be on the safer side the bank needs to address this as it will impact their profit. 


## Correlation check!


```python
cordata = bank_data.corr(method ='pearson')
cordata.style.background_gradient(cmap='summer')
```
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/corplot.PNG" alt="Cor Plot">


Age has some correlation (28.5%) with Churn rate. There are no real highly correlated features. NumberOfProducts and Balance are 30.4% Correlated among themselves. Apart from that there are no multi-collinear features which is good.

Let's try and create some features [feature engineering]. Let's See if we can create more useful features.

# Feature Engineering

One of the most important part in a data science/ml pipeline is the ability to create good features. Feature Engineering is the hard skill and this is where the creativity & knowledge of data scientist/ML practioner is required. 

## Train, Validation and Test Split

Before we create feature we will split the data into Train, CV and Test sets. Otherwise, there will this problem of data leakage. 

```python
# train cv test split - stratified sampling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y,random_state = 17)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train,random_state = 17)

# reset index
X_train = X_train.reset_index(drop = True)
X_cv = X_cv.reset_index(drop = True)
X_test = X_test.reset_index(drop = True)


print("train dimensions: ",X_train.shape, y_train.shape)
print("cv dimensions: ",X_cv.shape, y_cv.shape)
print("test dimensions: ",X_test.shape, y_test.shape)
```

    train dimensions:  (6400, 10) (6400,)
    cv dimensions:  (1600, 10) (1600,)
    test dimensions:  (2000, 10) (2000,)
    

**We will create features on train cv and test. Note: fit must only happen on train set.**


```python
# Balance and Salary Ratio
X_train['balance_salary_ratio'] = X_train['Balance'] / X_train['EstimatedSalary']
X_cv['balance_salary_ratio'] = X_cv['Balance'] / X_cv['EstimatedSalary']
X_test['balance_salary_ratio'] = X_test['Balance'] / X_test['EstimatedSalary']

# Does he have balance or not
X_train['balance_or_not'] = [0 if i == 0.0 else 1 for i in X_train['Balance']]
X_cv['balance_or_not'] = [0 if i == 0.0 else 1 for i in X_cv['Balance']]
X_test['balance_or_not'] = [0 if i == 0.0 else 1 for i in X_test['Balance']]

# CreditScore and Age -- if a young man has a high credit score?
X_train['creditscore_age_ratio'] = X_train['CreditScore'] / X_train['Age']
X_cv['creditscore_age_ratio'] = X_cv['CreditScore'] / X_cv['Age']
X_test['creditscore_age_ratio'] = X_test['CreditScore'] / X_test['Age']

# log feature?
X_train['creditscore_age_ratio_log'] = np.log10(X_train['creditscore_age_ratio'])
X_cv['creditscore_age_ratio_log'] = np.log10(X_cv['creditscore_age_ratio'])
X_test['creditscore_age_ratio_log'] = np.log10(X_test['creditscore_age_ratio'])

# Given his/her age does he/she have a better credit score
mean_age = np.mean(X_train['Age']) # use mean of train data for cv and test set
mean_credit = np.mean(X_train['CreditScore']) # use mean of train data for cv and test set

X_train['Better_Age_Credit'] = [1 if ((i < mean_age) and (j > mean_credit)) else 0 for i,j in zip(X_train['Age'],X_train['CreditScore'])]
X_cv['Better_Age_Credit'] = [1 if ((i < mean_age) and (j > mean_credit)) else 0 for i,j in zip(X_cv['Age'],X_cv['CreditScore'])]
X_test['Better_Age_Credit'] = [1 if ((i < mean_age) and (j > mean_credit)) else 0 for i,j in zip(X_test['Age'],X_test['CreditScore'])]


# does the customer hold a better age to credit ratio and an active customer
X_train['Better_Age_Credit_Active'] = [1 if ((i == 1) and (j == 1)) else 0 for i,j in zip(X_train['Better_Age_Credit'],X_train['IsActiveMember'])]
X_cv['Better_Age_Credit_Active'] = [1 if ((i == 1) and (j == 1)) else 0 for i,j in zip(X_cv['Better_Age_Credit'],X_cv['IsActiveMember'])]
X_test['Better_Age_Credit_Active'] = [1 if ((i == 1) and (j == 1)) else 0 for i,j in zip(X_test['Better_Age_Credit'],X_test['IsActiveMember'])]


# does he have multiple products
X_train['multi_products'] = [1 if i > 1 else 0 for i in X_train['NumOfProducts']]
X_cv['multi_products'] = [1 if i > 1 else 0 for i in X_cv['NumOfProducts']]
X_test['multi_products'] = [1 if i > 1 else 0 for i in X_test['NumOfProducts']]

# valuable customer? Better_Age_Credit and having more than 1 product?
mode_products = X_train['NumOfProducts'].mode()[0] # mode of train set 

X_train['Valuable_customer'] = [1 if ((i == 1) and (j > mode_products)) else 0  for i,j in zip(X_train['Better_Age_Credit_Active'],X_train['NumOfProducts'])]
X_cv['Valuable_customer'] = [1 if ((i == 1) and (j > mode_products)) else 0  for i,j in zip(X_cv['Better_Age_Credit_Active'],X_cv['NumOfProducts'])]
X_test['Valuable_customer'] = [1 if ((i == 1) and (j > mode_products)) else 0  for i,j in zip(X_test['Better_Age_Credit_Active'],X_test['NumOfProducts'])]

# Tenure and Age -- is he there from his/her young age?
X_train['tenure_age_ratio'] = X_train['Tenure'] / X_train['Age']
X_cv['tenure_age_ratio'] = X_cv['Tenure'] / X_cv['Age']
X_test['tenure_age_ratio'] = X_test['Tenure'] / X_test['Age']

# higher salary compared to his/her age?
mean_salary = np.mean(X_train['EstimatedSalary']) # mean sestimated alary of train set
X_train['high_salary_age'] = [1 if (i > mean_salary and j < mean_age) else 0 for i,j in zip(X_train['EstimatedSalary'],X_train['Age'])]
X_cv['high_salary_age'] = [1 if (i > mean_salary and j < mean_age) else 0 for i,j in zip(X_cv['EstimatedSalary'],X_cv['Age'])]
X_test['high_salary_age'] = [1 if (i > mean_salary and j < mean_age) else 0 for i,j in zip(X_test['EstimatedSalary'],X_test['Age'])]

print("New features created!")
X_train.head(3)

```

    New features created!
    
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>balance_salary_ratio</th>
      <th>balance_or_not</th>
      <th>creditscore_age_ratio</th>
      <th>creditscore_age_ratio_log</th>
      <th>Better_Age_Credit</th>
      <th>Better_Age_Credit_Active</th>
      <th>multi_products</th>
      <th>Valuable_customer</th>
      <th>tenure_age_ratio</th>
      <th>high_salary_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>640</td>
      <td>Spain</td>
      <td>Male</td>
      <td>43</td>
      <td>9</td>
      <td>172478.156250</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>191084.406250</td>
      <td>0.902628</td>
      <td>1</td>
      <td>14.883721</td>
      <td>1.172712</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.209302</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>850</td>
      <td>France</td>
      <td>Female</td>
      <td>24</td>
      <td>6</td>
      <td>0.000000</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>13159.900391</td>
      <td>0.000000</td>
      <td>0</td>
      <td>35.416667</td>
      <td>1.549208</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>494</td>
      <td>Germany</td>
      <td>Female</td>
      <td>38</td>
      <td>7</td>
      <td>174937.640625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>40084.320312</td>
      <td>4.364241</td>
      <td>1</td>
      <td>13.000000</td>
      <td>1.113943</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.184211</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>

We have created some features, but we don't know how useful it will be. We will check their influence at the time of creating Models.

Also note that ratio features like balance_salary_ratio, creditscore_age_ratio, etc may create multi-collinearity.

Note: It has been stated that multi-collinearity is not an issue when using sckit-learn models. [Source](https://www.linkedin.com/posts/justmarkham_sklearntips-machinelearning-python-activity-6651812212270788609-lhE1/)

# Data Preprocessing

Preparing data for Model building.! Encoding categorical data and normalizing numerical data.

### Encoding Categorical data: Gender and Geography

```python
# Encoding Gender
X_train['Gender'] = X_train['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
X_cv['Gender'] = X_cv['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
X_test['Gender'] = X_test['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
```


```python
# One Hot Encoding - Geography
from sklearn.preprocessing import OneHotEncoder

# left-to-right column order is alphabetical (France, Germany, Spain)
ohe = OneHotEncoder(sparse=False)
X_train_geo_ohe = pd.DataFrame(ohe.fit_transform(X_train[['Geography']]),columns = ['France', 'Germany', 'Spain'])
X_cv_geo_ohe = pd.DataFrame(ohe.transform(X_cv[['Geography']]),columns = ['France', 'Germany', 'Spain'])
X_test_geo_ohe = pd.DataFrame(ohe.transform(X_test[['Geography']]),columns = ['France', 'Germany', 'Spain'])

# drop the Geography column
X_train.drop('Geography',axis=1,inplace = True)
X_cv.drop('Geography',axis=1,inplace = True)
X_test.drop('Geography',axis=1,inplace = True)

# Concat the One Hot encoded columns
X_train = pd.concat([X_train, X_train_geo_ohe],axis = 1)
X_cv = pd.concat([X_cv, X_cv_geo_ohe],axis = 1)
X_test = pd.concat([X_test, X_test_geo_ohe],axis = 1)

```
### Data Standardization

Each feature is of different scales/units. Features like Salary and Balance have higher range of values compared to Age, Tenure. We need to standardise features before feeding them into our Models. 


```python
from sklearn.preprocessing import StandardScaler
# features to standardise
cols_norm = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','balance_salary_ratio',
            'creditscore_age_ratio','creditscore_age_ratio_log','tenure_age_ratio']

sc = StandardScaler()
sc.fit(X_train[cols_norm]) # fit has to happen only on train set

X_train[cols_norm] = sc.transform(X_train[cols_norm])
X_cv[cols_norm] = sc.transform(X_cv[cols_norm])
X_test[cols_norm] = sc.transform(X_test[cols_norm])

print("Standardized!")
X_train.head()
```

    Standardized!

# Model 

We will try multiple types of algorithms for this data.

  1. Logistic Regression
  2. XGBoost
  3. XGBoost with SMOTE Oversampling

## Logistic Regression

We will use sklearn's Logistic Regression with Hyperparameter Tuning

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
train_auc = []
cv_auc = []
C_list = [0.0001, 0.001 ,0.01 ,0.1, 1, 10, 100, 1000] # list of various inverse of lambda values we want to compare.
for i in tqdm(C_list): # for each values of C_list (i)
    lr = LogisticRegression(C=i,class_weight='balanced', random_state=17, solver = 'liblinear') # initialize Logistic Model with C = i
    lr.fit(X_train, y_train) # fit the lr model on the train data
    y_train_pred = batch_predict(lr, X_train) # Predict on the train data    
    y_cv_pred = batch_predict(lr, X_cv) # Predict on cross validation data
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs        
    train_auc.append(roc_auc_score(y_train,y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(C_list, train_auc, label='Train AUC')
plt.plot(C_list, cv_auc, label='CV AUC')
plt.xscale('log') # change the scale of x axis
plt.autoscale(True)
plt.scatter(C_list, train_auc, label='Train AUC points')
plt.scatter(C_list, cv_auc, label='CV AUC points')
plt.xscale('log')
plt.legend()
plt.xlabel("C: hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.grid()
plt.show()
```

    100%|██████████| 8/8 [00:00<00:00, 16.11it/s]

<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_53_1.png" alt="LR">

From the above image we can see that at C = 10 the train and cross validation AUC scores are high as well as close to each other. Therefore we can set the optimum C as 10.

After building the Logistic Regression model using C = 10 on train data, we get the AUC scores on train and test as shown below.
    
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/output_54_0.png" alt="LR Test">


    Logistic Regression - Test AUC:  0.836560751814989

## Logistic Regression Feature importances

    > In Logistic Regression when the absolute weights are large then w.T*x is also large. 

    > Weights of the feature indicate their importance. 
    
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weights</th>
      <th>Abs_Weights</th>
      <th>Features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>-5.567203</td>
      <td>5.567203</td>
      <td>multi_products</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-2.543284</td>
      <td>2.543284</td>
      <td>creditscore_age_ratio_log</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.352321</td>
      <td>2.352321</td>
      <td>NumOfProducts</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.474438</td>
      <td>1.474438</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.005711</td>
      <td>1.005711</td>
      <td>IsActiveMember</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.949018</td>
      <td>0.949018</td>
      <td>CreditScore</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.769292</td>
      <td>0.769292</td>
      <td>creditscore_age_ratio</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.733009</td>
      <td>0.733009</td>
      <td>Age</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.698971</td>
      <td>0.698971</td>
      <td>Valuable_customer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.605773</td>
      <td>0.605773</td>
      <td>Gender</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.581897</td>
      <td>0.581897</td>
      <td>Spain</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.533779</td>
      <td>0.533779</td>
      <td>France</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.498421</td>
      <td>0.498421</td>
      <td>high_salary_age</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.425378</td>
      <td>0.425378</td>
      <td>Better_Age_Credit</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.339650</td>
      <td>0.339650</td>
      <td>tenure_age_ratio</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.323940</td>
      <td>0.323940</td>
      <td>Tenure</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.173293</td>
      <td>0.173293</td>
      <td>EstimatedSalary</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.154712</td>
      <td>0.154712</td>
      <td>Better_Age_Credit_Active</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.092624</td>
      <td>0.092624</td>
      <td>balance_or_not</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.023764</td>
      <td>0.023764</td>
      <td>balance_salary_ratio</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.010782</td>
      <td>0.010782</td>
      <td>Balance</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.002682</td>
      <td>0.002682</td>
      <td>HasCrCard</td>
    </tr>
  </tbody>
</table>
</div>

From the absolute weights we can see features like 'multi_products','creditscore_age_ratio_log','NumOfProducts','Germany', 'IsActiveMember' are most important and features like 'HasCrCard','Balance', and 'balance_salary_ratio' are least important. 

So feature engineering has helped alot.

Let's drop the least important features and build the model again.
