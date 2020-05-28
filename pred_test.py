import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


dat = pd.read_csv('fifa_cleaned.csv')



#feature_cols = ['age', 'height_cm', 'weight_kgs', 'overall_rating', 'potential','international_reputation(1-5)', 'weak_foot(1-5)', 'skill_moves(1-5)', 'crossing', 'finishing', 'heading_accuracy',	'short_passing', 'volleys','dribbling','curve',	'freekick_accuracy', 'long_passing', 'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance','shot_power','jumping', 'stamina', 'strength', 'long_shots', 'aggression' , 'interceptions', 'positioning', 'vision', 'penalties', 'composure',	'marking', 'standing_tackle', 'sliding_tackle',	'GK_diving', 'GK_handling',	'GK_kicking', 'GK_positioning',	'GK_reflexes']
feature_cols = ['age']


X = dat[feature_cols]
Y = dat.value_euro


print(X.head(5))
'''
nan_values = X.isna()
nan_columns = nan_values.any()

columns_with_nan = X.columns[nan_columns].tolist()
print(columns_with_nan)
'''

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

y = Y.values #returns a numpy array
y_scaled = min_max_scaler.fit_transform(y)
y_s = pd.DataFrame(y_scaled)



print(df)





X_train, X_test, y_train, y_test = train_test_split(df,y_s, random_state = 1)

lin = LinearRegression()
lin.fit(X_train,y_train)

'''


rand_clf = RandomForestRegressor(n_estimators=1500, max_features='sqrt')

rand_clf.fit(X_train, y_train)
y_pred = rand_clf.predict(X_test)

print("MAE : %f" % metrics.mean_absolute_error(y_test, y_pred))
print("MSE : %f" % metrics.mean_squared_error(y_test, y_pred))
print("RMSE : %f" % np.sqrt(metrics.mean_squared_error(y_test, y_pred)))'''