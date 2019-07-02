import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# build a dataset
dataset = pd.read_csv('weather.csv')

# how much data
print(dataset.shape)

# data info
print(dataset.describe())

# graph bw maxTemp and MinTemp
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.savefig('graph_maxVsmin.png')

# graph: avgMaxTemp
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.savefig('graph_avgMaxTemp')

# graph: avgMinTemp
plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['MinTemp'])
plt.savefig('graph_avgMinTemp')

# Data Splicing

x = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

# split dataset into training data and testing data
# Training: 80% and Testing: 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# training algorithm
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# data from linear regression

print('Intercept: ', regressor.intercept_)
print('Coefficient: ', regressor.coef_)

# predict data
# predict: predefined func , pass test data independent variable

y_pred = regressor.predict(x_test)

# compare actual and predicted

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# graph bw actual and predicted
# bar
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', color='green')
plt.grid(which='minor', color='black')
plt.savefig('actual_pred_bar')

# line
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred, color='red')
plt.savefig('actualVspred')

# errors
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root mean squared error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
