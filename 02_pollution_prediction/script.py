import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

## load the dataset from the csv 
data = pd.read_csv('../02_pollution_prediction/AirQualityUCI.csv')


## transform the dataset and combine date and time with datetime data type
data.drop(['CO_level'],axis=1,inplace=True)
data['DateTime'] = pd.to_datetime(data['Date']+' '+data['Time'], errors='coerce')
data.drop(['Date','Time'],axis=1, inplace=True)
data.set_index('DateTime', inplace=True)

data['hour'] = data.index.hour
data['day'] = data.index.day
data['month'] = data.index.month
data['weekday'] = data.index.weekday

# interactive features
data['NO2_Nox'] = data['NO2_GT'] * data['Nox_GT']
data['CO_NO2_ratio'] = data['CO_GT'] / (data['NO2_GT'] + 1)
data['NO2_GT_sq'] = data['NO2_GT'] ** 2
data['Nox_GT_sq'] = data['Nox_GT'] ** 2
data['T_RH'] = data['T'] * data['RH']

# Drop rows with NaN due to lag/rolling
data.dropna(inplace=True)

corr = data.corr()['CO_GT'].sort_values(ascending=False)
print(corr)


## split the dataset into independent and dependent variables
X = data.drop(['CO_GT'],axis=1)
y = data['CO_GT']

## train and test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.245,random_state=42)

## scale the training data 
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## load the linear regression model
regression = LinearRegression()
regression.fit(X_train,y_train)

## perform cross-validation
mse = cross_val_score(regression, X_train,y_train,cv=10,scoring='neg_mean_squared_error')
print("Cross-validation MSE (negative values):", mse)
print("Average MSE:", -np.mean(mse))

## make predictions
y_pred = regression.predict(X_test)

## plot the difference
sns.displot(y_pred-y_test,kind='kde')
plt.show()

# Make sure y_test and y_pred are aligned with DateTime
y_test_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

# Resample daywise (mean per day)
daily = y_test_df.resample('D').mean()

# Plot daywise Actual vs Predicted
plt.figure(figsize=(15,5))
plt.plot(daily.index, daily['Actual'], label='Actual CO_GT', color='green')
plt.plot(daily.index, daily['Predicted'], label='Predicted CO_GT', color='red')
plt.xlabel("Date")
plt.ylabel("CO Concentration")
plt.title("Daily Average CO_GT: Actual vs Predicted")
plt.legend()
plt.show()

## performance metrics 

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test MSE :", mse)
print("Test RMSE:", rmse)
print("Test MAE :", mae)
print("Test R² Score:", r2)

   




