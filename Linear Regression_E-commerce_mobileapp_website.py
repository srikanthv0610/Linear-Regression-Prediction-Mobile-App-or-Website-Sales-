#A Linear Regression project
#trying to predict whether e-commerce company should go for mobile app or website sales

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Ecommerce Customers")


print(customers.head())

print(customers.describe())

print(customers.info())

#Use seaborn to create a jointplot to compare the different variables with Yearly Amount Spent columns.

#Important to Note: More time on site, more money spent.

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)

#Using pairplot to compare our datasets:

sns.pairplot(customers)

#Create a linear model plot (using seaborn's lmplot) of
#Yearly Amount Spent vs. Length of Membership.

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

#Training and Testing Data
#Now that we've explored the data a bit, let's go ahead and split the data into
#training and testing sets. Set a variable X equal to the numerical features of
#the customers and a variable y equal to the "Yearly Amount Spent" column.


y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


#Use model_selection.train_test_split from sklearn to split the data into
#training and testing sets. Set test_size=0.3

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

#Training the Model

#Import LinearRegression from sklearn.linear_model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

#Train/fit lm on the training data.
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


#Print out the coefficients of the model
# The coefficients
print('Coefficients: \n', lm.coef_)
#Coefficients:
# [ 25.287716  39.171783   0.321668  61.248401]


#Predicting Test Data:


predictions = lm.predict(X_test)

#Create a scatterplot of the real test values versus the predicted values.

plt.figure()
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Evaluating the Model

#Evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2)

#Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error.

# calculate the metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#>> Result:
#MAE: 8.414860085645259
#MSE: 109.27910351983392
#RMSE: 10.45366459763436

#Residuals

#Plot a histogram of the residuals and make sure it looks normally distributed.

plt.figure()
sns.distplot((y_test-predictions),bins=50)
plt.show()

#Conclusion

#Do we focus our efforst on mobile app or website development?
#Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.

#Recreate the dataframe below.

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

#                      Coeffecient
#Avg. Session Length     25.287716
#Time on App             39.171783
#Time on Website          0.321668
#Length of Membership    61.248401

#How can you interpret these coefficients?

#Interpreting the coefficients:

# 1 unit increase in Avg. Session Length is associated with an increase of 25.28 total euros spent, keeping all other features fixed.
# 1 unit increase in Time on App is associated with an increase of 39.17 total euros spent, keeping all other features fixed.
# 1 unit increase in Time on Website is associated with an increase of 0.32 total euros spent, keeping all other features fixed.
# 1 unit increase in Length of Membership is associated with an increase of 61.25 total euros spent, keeping all other features fixed.


