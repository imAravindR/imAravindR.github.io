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




<style  type="text/css" >
    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col0 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col1 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col2 {
            background-color:  #078366;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col3 {
            background-color:  #3c9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col4 {
            background-color:  #3e9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col5 {
            background-color:  #028066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col6 {
            background-color:  #289366;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col7 {
            background-color:  #028066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col8 {
            background-color:  #1c8e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col0 {
            background-color:  #058266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col1 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col2 {
            background-color:  #048266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col3 {
            background-color:  #41a066;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col4 {
            background-color:  #359a66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col5 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col6 {
            background-color:  #359a66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col7 {
            background-color:  #018066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col8 {
            background-color:  #61b066;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col0 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col1 {
            background-color:  #058266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col2 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col3 {
            background-color:  #399c66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col4 {
            background-color:  #3e9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col5 {
            background-color:  #098466;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col6 {
            background-color:  #1c8e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col7 {
            background-color:  #048266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col8 {
            background-color:  #1f8f66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col0 {
            background-color:  #088466;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col1 {
            background-color:  #0e8666;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col2 {
            background-color:  #048266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col3 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col4 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col5 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col6 {
            background-color:  #209066;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col7 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col8 {
            background-color:  #3c9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col0 {
            background-color:  #098466;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col1 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col2 {
            background-color:  #0a8466;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col3 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col4 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col5 {
            background-color:  #048266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col6 {
            background-color:  #249266;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col7 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col8 {
            background-color:  #178b66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col0 {
            background-color:  #058266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col1 {
            background-color:  #048266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col2 {
            background-color:  #0c8666;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col3 {
            background-color:  #389c66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col4 {
            background-color:  #3c9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col5 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col6 {
            background-color:  #1f8f66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col7 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col8 {
            background-color:  #209066;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col0 {
            background-color:  #0d8666;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col1 {
            background-color:  #1c8e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col2 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col3 {
            background-color:  #399c66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col4 {
            background-color:  #3d9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col5 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col6 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col7 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col8 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col0 {
            background-color:  #068266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col1 {
            background-color:  #058266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col2 {
            background-color:  #088466;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col3 {
            background-color:  #3e9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col4 {
            background-color:  #3e9e66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col5 {
            background-color:  #018066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col6 {
            background-color:  #209066;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col7 {
            background-color:  #ffff66;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col8 {
            background-color:  #259266;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col0 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col1 {
            background-color:  #4ea666;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col2 {
            background-color:  #038166;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col3 {
            background-color:  #52a866;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col4 {
            background-color:  #329866;
            color:  #000000;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col5 {
            background-color:  #018066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col6 {
            background-color:  #008066;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col7 {
            background-color:  #058266;
            color:  #f1f1f1;
        }    #T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col8 {
            background-color:  #ffff66;
            color:  #000000;
        }</style><table id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43a" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CreditScore</th>        <th class="col_heading level0 col1" >Age</th>        <th class="col_heading level0 col2" >Tenure</th>        <th class="col_heading level0 col3" >Balance</th>        <th class="col_heading level0 col4" >NumOfProducts</th>        <th class="col_heading level0 col5" >HasCrCard</th>        <th class="col_heading level0 col6" >IsActiveMember</th>        <th class="col_heading level0 col7" >EstimatedSalary</th>        <th class="col_heading level0 col8" >Exited</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row0" class="row_heading level0 row0" >CreditScore</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col0" class="data row0 col0" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col1" class="data row0 col1" >-0.003965</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col2" class="data row0 col2" >0.000842</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col3" class="data row0 col3" >0.006268</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col4" class="data row0 col4" >0.012238</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col5" class="data row0 col5" >-0.005458</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col6" class="data row0 col6" >0.025651</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col7" class="data row0 col7" >-0.001384</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow0_col8" class="data row0 col8" >-0.027094</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row1" class="row_heading level0 row1" >Age</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col0" class="data row1 col0" >-0.003965</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col1" class="data row1 col1" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col2" class="data row1 col2" >-0.009997</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col3" class="data row1 col3" >0.028308</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col4" class="data row1 col4" >-0.030680</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col5" class="data row1 col5" >-0.011721</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col6" class="data row1 col6" >0.085472</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col7" class="data row1 col7" >-0.007201</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow1_col8" class="data row1 col8" >0.285323</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row2" class="row_heading level0 row2" >Tenure</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col0" class="data row2 col0" >0.000842</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col1" class="data row2 col1" >-0.009997</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col2" class="data row2 col2" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col3" class="data row2 col3" >-0.012254</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col4" class="data row2 col4" >0.013444</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col5" class="data row2 col5" >0.022583</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col6" class="data row2 col6" >-0.028362</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col7" class="data row2 col7" >0.007784</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow2_col8" class="data row2 col8" >-0.014001</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row3" class="row_heading level0 row3" >Balance</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col0" class="data row3 col0" >0.006268</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col1" class="data row3 col1" >0.028308</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col2" class="data row3 col2" >-0.012254</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col3" class="data row3 col3" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col4" class="data row3 col4" >-0.304180</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col5" class="data row3 col5" >-0.014858</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col6" class="data row3 col6" >-0.010084</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col7" class="data row3 col7" >0.012797</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow3_col8" class="data row3 col8" >0.118533</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row4" class="row_heading level0 row4" >NumOfProducts</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col0" class="data row4 col0" >0.012238</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col1" class="data row4 col1" >-0.030680</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col2" class="data row4 col2" >0.013444</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col3" class="data row4 col3" >-0.304180</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col4" class="data row4 col4" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col5" class="data row4 col5" >0.003183</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col6" class="data row4 col6" >0.009612</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col7" class="data row4 col7" >0.014204</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow4_col8" class="data row4 col8" >-0.047820</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row5" class="row_heading level0 row5" >HasCrCard</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col0" class="data row5 col0" >-0.005458</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col1" class="data row5 col1" >-0.011721</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col2" class="data row5 col2" >0.022583</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col3" class="data row5 col3" >-0.014858</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col4" class="data row5 col4" >0.003183</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col5" class="data row5 col5" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col6" class="data row5 col6" >-0.011866</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col7" class="data row5 col7" >-0.009933</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow5_col8" class="data row5 col8" >-0.007138</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row6" class="row_heading level0 row6" >IsActiveMember</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col0" class="data row6 col0" >0.025651</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col1" class="data row6 col1" >0.085472</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col2" class="data row6 col2" >-0.028362</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col3" class="data row6 col3" >-0.010084</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col4" class="data row6 col4" >0.009612</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col5" class="data row6 col5" >-0.011866</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col6" class="data row6 col6" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col7" class="data row6 col7" >-0.011421</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow6_col8" class="data row6 col8" >-0.156128</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row7" class="row_heading level0 row7" >EstimatedSalary</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col0" class="data row7 col0" >-0.001384</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col1" class="data row7 col1" >-0.007201</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col2" class="data row7 col2" >0.007784</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col3" class="data row7 col3" >0.012797</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col4" class="data row7 col4" >0.014204</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col5" class="data row7 col5" >-0.009933</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col6" class="data row7 col6" >-0.011421</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col7" class="data row7 col7" >1.000000</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow7_col8" class="data row7 col8" >0.012097</td>
            </tr>
            <tr>
                        <th id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43alevel0_row8" class="row_heading level0 row8" >Exited</th>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col0" class="data row8 col0" >-0.027094</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col1" class="data row8 col1" >0.285323</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col2" class="data row8 col2" >-0.014001</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col3" class="data row8 col3" >0.118533</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col4" class="data row8 col4" >-0.047820</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col5" class="data row8 col5" >-0.007138</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col6" class="data row8 col6" >-0.156128</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col7" class="data row8 col7" >0.012097</td>
                        <td id="T_a4ff6a28_ef42_11ea_a883_54e1ad44a43arow8_col8" class="data row8 col8" >1.000000</td>
            </tr>
    </tbody></table>


