import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

data = pd.read_csv("iot_telemetry_data.csv")
data.head()
data.replace(['b8:27:eb:bf:9d:51', '00:0f:00:70:91:0a', '1c:bf:ce:15:ec:4d'], ['C1','C2','C3'], inplace=True)
data.head()
data['time'] = pd.to_datetime(data['ts'])
data['time']
data_1 = data[data.device == 'C1']
data_2 = data[data.device == 'C2']
data_3 = data[data.device == 'C3']

data_1
data_2
data_3
plt.plot(data_1['time'], data_1['co'], label='Device_C1')
plt.plot(data_2['time'], data_2['co'], label='Device_C2')
plt.plot(data_3['time'], data_3['co'], label='Device_C3')
plt.legend()
plt.show()

plt.plot(data_1['time'], data_1['humidity'], label='Device_C1')
plt.plot(data_2['time'], data_2['humidity'],label='Device_C2')
plt.plot(data_3['time'], data_3['humidity'],label='Device_C3')
plt.legend()
plt.show()

df['ds'] = df['ts'].apply(lambda x: str(datetime.fromtimestamp(x)))
df['ds'] = df['ds'].apply(lambda x: x.split(".")[0])
df['y'] = df['humidity']

df.drop(['ts', 'light', 'motion', 'device', 'co', 'humidity', 'smoke', 'temp', 'lpg'], axis=1, inplace=True)
df.head()

rolling_mean = df.rolling(7).mean()
rolling_std = df.rolling(7).std()
from statsmodels.tsa.stattools import adfuller
adft = adfuller(df,autolag="AIC")
from statsmodels.tsa.seasonal import seasonal_decompose


decompose = seasonal_decompose(df['time'],model='humidity', period=7)
decompose.plot()
plt.show()

df['humidity'] = df.index
train = df[df['time'] < pd.to_datetime()]
train['train'] = train['humidity']
del train['time']
del train['humidity']
test = df[df['Date'] >= pd.to_datetime()]
del test['time']
test['test'] = test['humiditys']
del test['humidity']
plt.plot(train, color = "black")
plt.plot(test, color = "red")
sns.set()
plt.show()


from pmdarima.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])