##############################################
# IMPORT LIBRARIES #
# #############################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api  as sm
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.tsa.holtwinters as ets
from TS_toolbox import DSD as dsd
from TS_toolbox import Models as m
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


##############################################
#	READING THE DATA	#
#############################################

data =  pd.read_csv('https://raw.githubusercontent.com/hemangikinger/occupancy-dataset/main/occupancy_data/datatraining.txt')
print('Checking if any nan values are present')
print(data.isna().any())
data['date'] = pd.to_datetime(data['date'])
df1 = data.copy()

##############################################
#	PLOTTING THE TARGET VARIABLE	#
#############################################

dsd.plot_graph(data, 'Temperature', 'date')

##############################################
#	ACF/ PACF plot 	#
#############################################

dsd.ACF_PACF_Plot(data['Temperature'],50,'Temperature')

##############################################
#	Heatmap 	#
#############################################

corr_df = data.corr(method='pearson')
ax = sns.heatmap(corr_df, annot = True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

##############################################
#	DATA SPLITING 	#
#############################################

df_train, df_test = train_test_split(data, shuffle=False, test_size=0.20)

##############################################
#	STATIONARITY 	#
#############################################

dsd.Plot_Rolling_Mean_Var(data,'Temperature')

print('---ADF Test for Temperature---')
dsd.ADF_Cal(df_train['Temperature'].values)

print('---KPSS Test for Temperature---')
dsd.kpss_test(df_train['Temperature'].values)

df_train['Temperature_diff'] = df_train['Temperature'] - df_train['Temperature'].shift(1)

dsd.plot_graph(df_train, 'Temperature_diff', 'date')


dsd.ACF_PACF_Plot(df_train['Temperature_diff'].dropna(),50,'Temperature_diff')

dsd.Plot_Rolling_Mean_Var(df_train,'Temperature_diff')

print('---ADF Test for Temperature_diff---')
dsd.ADF_Cal(df_train['Temperature_diff'].dropna().values)

print('---KPSS Test for Temperature_diff---')
dsd.kpss_test(df_train['Temperature_diff'].dropna().values)


##############################################
#	TIME SERIES DECOMPOSITION 	#
#############################################

dsd.ts_decomposition(data,'Temperature')
dsd.ts_decomposition(df_train,'Temperature_diff')

# ##############################################
# #	FEATURE SELECTION	#
# ##############################################

x = df1.drop(['Temperature','date'], axis=1)
y = df1['Temperature']
x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, shuffle=False, test_size=0.20)
sc = StandardScaler()

X_train = sc.fit_transform(x_train1)
X_test = sc.transform(x_test1)

pca = PCA(n_components = 'mle')

X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

s,d,v = np.linalg.svd (X_train, full_matrices = True)

print(f'singular values of x are {d}')
print(f'The condition number for x is {LA.cond(X_train)}')


X_train = sm.add_constant(X_train, prepend=False)
model = sm.OLS(y_train1, X_train).fit()
print(model.summary())

print("t-test p-values for all features: \n", model.pvalues)

print("#" * 100)

print("F-test for final model: \n", model.f_pvalue)


prediction = model.predict(X_train)


##############################################
#	HOLT WINTERS METHOD	#
##############################################

x = df1.drop(['Temperature','date'], axis=1)
y = df1['Temperature']
x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, shuffle=False, test_size=0.20)


m.holt_winters_method(df_train['Temperature'].values, df_test['Temperature'].values)

##############################################
#	Base Models	#
##############################################

m.average_forecast_method(df_train['Temperature'], df_test['Temperature'])


m.naive_method(df_train['Temperature'].values, df_test['Temperature'].values)


m.drift_method(df_train['Temperature'].values, df_test['Temperature'].values)

m.ses_method(df_train['Temperature'].values, df_test['Temperature'].values, 0.5)

m.holt_linear_method(df_train['Temperature'].values, df_test['Temperature'].values)


#############################################
	# GPAC	#
#############################################

g = df_train['Temperature_diff'].dropna()
dsd.GCPA_Cal(g)

##############################################
#	ARMA	#
##############################################

y_train = df_train['Temperature'].values
y_test = df_test['Temperature'].values


m.ARIMA_method(2,0, 1, 20, y_train, y_test)

m.ARIMA_method(1,0, 1, 20, y_train, y_test)

m.ARIMA_method(0,1, 1, 20, y_train, y_test)

    # (0,1,1) - (0,1,1,1440)

# adb = [(0, 0, 0),  (0, 0, 1),  (0, 1, 0),  (0, 1, 1),  (1, 0, 0),  (1, 0, 1), (1, 1, 0),  (1, 1, 1), (2,1,0)]
# # Seasonal Order Combination
# ADB = [(0, 0, 0, 1440),  (0, 0, 1, 1440),  (0, 1, 0, 1440),  (0, 1, 1, 1440),  (1, 0, 0, 1440),  (1, 0, 1, 1440),  (1, 1, 0, 1440),  (1, 1, 1, 1440), (2,1,0),1440]
#
# for i in adb:
#     for j in ADB:
#         mod = sm.tsa.statespace.SARIMAX(df_train['Temperature'], order=i, seasonal_param_order=j)
#         res = mod.fit()
#
#         print(f'SARIMA{i}x{j} - AIC:{res.aic}')