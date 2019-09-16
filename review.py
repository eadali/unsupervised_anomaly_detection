import pandas
from matplotlib import pyplot
from numpy import int64, zeros, array, where, clip, log, copy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
    
    

# CONFIGS
# =============================================================================
# Sets Isolation Forest parameters
rf_contamination = 0.01

# Sets OneClassSVM parameters
ocsvm_nu = 0.001
ocsvm_gamma = 0.01
# =============================================================================



# LOAD DATAFRAME PREPROCESS
# =============================================================================
# Loads csv file
dataframe = pandas.read_csv('./machine_temperature_system_failure.csv')

# Converts argument to datetime.
date = pandas.to_datetime(dataframe['timestamp'])
timestamp = date.astype(int64)

dataframe['timestamp'] = (timestamp-min(timestamp)) / (max(timestamp)-min(timestamp))

# Gets hour value
dataframe['hour'] = date.dt.hour

# Gets daylight or night
dataframe['daylight'] = ((dataframe['hour'] >= 7) & (dataframe['hour'] <= 22)).astype(int)

# Gets day value
dataframe['day_of_week'] = date.dt.dayofweek

# Gets weekday or weekend
dataframe['weekday'] = (dataframe['day_of_week'] < 5).astype(int)
# =============================================================================



# ISOLATION FOREST 
# =============================================================================
# Preprocess data for model
x_data = dataframe[['value', 'hour', 'daylight', 'day_of_week', 'weekday']]
x_data = StandardScaler().fit_transform(x_data)

# Trains IsolationForest  and predicts
iforest =  IsolationForest(behaviour='new', contamination=rf_contamination).fit(x_data)
y_pred = iforest.predict(x_data) < 0.1

# Plots prediction result
iforest_fig = pyplot.figure()
ax = iforest_fig.add_subplot(111)
ax.plot(dataframe['timestamp'], dataframe['value'])
ax.set_title('Isolation Forest')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
ax.grid()

for x_pos in dataframe['timestamp'][dataframe['failure']]:
    ax.axvspan(x_pos-0.016, x_pos+0.016, color='orange', linewidth=2)
    
ax.scatter(dataframe['timestamp'][y_pred], dataframe['value'][y_pred],
           color='red', s=64, zorder=5)
# =============================================================================



# ONE CLASS SVM
# =============================================================================
# Preprocess data for model
x_data = dataframe[['value', 'hour', 'daylight', 'day_of_week', 'weekday']]
x_data = StandardScaler().fit_transform(x_data)

# Trains OneClassSVM  and predicts
ocv =  OneClassSVM(nu=ocsvm_nu, gamma=ocsvm_gamma).fit(x_data)
y_pred = ocv.predict(x_data) < 0.1

# Plots prediction result
svm_fig = pyplot.figure()
ax = svm_fig.add_subplot(111)
ax.plot(dataframe['timestamp'], dataframe['value'])
ax.set_title('One Class SVM')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature')
ax.grid()

for x_pos in dataframe['timestamp'][dataframe['failure']]:
    ax.axvspan(x_pos-0.016, x_pos+0.016, color='orange', linewidth=2)
    
ax.scatter(dataframe['timestamp'][y_pred], dataframe['value'][y_pred],
           color='red', s=64, zorder=5)
# =============================================================================

pyplot.show()    