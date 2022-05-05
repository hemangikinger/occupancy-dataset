
# This file contains the functionsa required to perform
# 1. Description
# 2. Stationarity
# 3. Time Series Decomposition

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api  as sm
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
# import statsmodels.tsa.holtwinters as ets

def add(a,b):
    print(a+b)

def plot_graph(df,target_col,time_col):
    plt.figure()
    df.plot(x = time_col, y = target_col)
    plt.title(f'Trend of {target_col} ')
    plt.xlabel(time_col)
    plt.ylabel(target_col)
    plt.grid()
    plt.tight_layout()
    plt.show()

def ACF_PACF_Plot(y, lags, l):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags, title = f'Autocorrelation of {l}')
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags, title = f'Partial-autocorrelation of {l}')
    fig.tight_layout(pad=3)
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

def autocorrelation(lags,list):
    val = [1]
    lag = [0]
    den = 0
    num = 0
    for k in range (0,len(list)):
        d = (list[k] - list.mean())**2
        den+= d
    for i in range(1,lags+1):
        num = 0
        for j in range (i,len(list)):
            n = ((list[j] - list.mean()) * (list[abs(j-i)]- list.mean()))
            num+= n
        cor = num/den
        val+=[cor]
        lag += [i]
    lag = np.array(lag)
    val = np.array(val)
    vals = val
    vals = vals[vals != 1]
    val1 = val[1:]
    lag1 = np.negative(lag[1:])
    val = np.concatenate((val1[::-1],val), axis=None)
    lag = np.concatenate((lag1[::-1], lag), axis=None)
    val = val[val != -1]
    return val,lag,vals

def Cal_rolling_mean_var(df, col):
    col_name_rm = 'rm_'+col
    col_name_var = 'var_' + col
    df[col_name_rm] = df[col]
    df[col_name_var] = df[col]
    for i in range (1,len(df)):
        if i == 1:
            df.iloc[i, df.columns.get_loc(col_name_rm)] = df[col][i]
            df.iloc[i, df.columns.get_loc(col_name_var)] = 0
        else:
            df.iloc[i,df.columns.get_loc(col_name_rm)] = df.head(i+1)[col].mean()
            df.iloc[i, df.columns.get_loc(col_name_var)] = df.head(i+1)[col].var()

    return col_name_rm, col_name_var

def Plot_Rolling_Mean_Var (df, l):

    rollingmean, rollingvar  = Cal_rolling_mean_var(df, l)

    plt.figure(figsize=(50,50))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(df[rollingmean], label = f'Rolling Mean of {l}')
    ax1.legend()
    ax1.set_title(f'Rolling mean of {l}')
    ax1.set_xlabel('Time(days)')
    ax1.set_ylabel(f'Rolling mean of {l}')
    ax1.grid()
    ax2.plot(df[rollingvar], label = f'Rolling Variance of {l} ')
    ax2.legend()
    ax2.set_title(f'Rolling variance of {l} ')
    ax2.set_xlabel('Time(days)')
    ax2.set_ylabel(f'Rolling variance of {l}')
    ax2.grid()
    plt.tight_layout(pad = 3)
    plt.show()


def ts_decomposition(df,col):

    res = STL(df[col].dropna(), period=1440).fit()

    fig = res.plot()
    # plt.title(f'STL Decomposition of {col}')
    plt.show()

    T = res.trend
    S = res.seasonal
    R = res.resid

    adj_seasonal = df[col].dropna() - S

    plt.figure()
    plt.plot(df[col].dropna(), label='Original Data')
    plt.plot(adj_seasonal, label='Adjusted Seasonal')
    plt.title(f'Seasonally Adjusted Data vs Original Data of {col}')
    plt.xlabel('Frequency')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    detrended = df[col].dropna() - T

    plt.figure()
    plt.plot(df[col].dropna(), label='Original Data')
    plt.plot(detrended, label='Detrended Data')
    plt.title(f'Detrended Data vs Original Data of {col}')
    plt.xlabel('Frequency')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(S) + np.array(R)))

    print(f"Strength of seasonality of {col} is {F}")

    F = np.maximum(0, 1 - np.var(np.array(R))/np.var(np.array(T) + np.array(R)))

    print(f"Strength of trend of {col} is {F}")


def GCPA_Cal(val):

    # acf = np.arange(len(acf1))
    val  = np.array(val[1:])

    acf, _, _ = autocorrelation(15, val)

    ki = 7
    ji = 7

    arr = np.zeros((7, 7))

    for k in range(1, ki):
        for j in range(0, ji):
            num = np.zeros((k, k))
            den = np.zeros((k, k))
            n1 = np.zeros((k, 1))

            if k == 1 :
                for h in range(0,k):
                    num[0][h] = acf[j+h+1]
                    den[0][h] = acf[j-k+h+1]
            else:
                for x in range(0,k):
                    for y in range (0,k):
                        den[x][y] = acf[abs(j-y+x)]
                num = den[:, :-1]

                for h in range(0,k):
                    n1[h][0] = acf[j+h+1]

                num = np.append(num, n1, 1)

            # print(num)
            # print(den)
            # print('-----')


            num_d = np.linalg.det(num)
            den_d = np.linalg.det(den)
            # print(num_d)
            # print(den_d)
            # print('-----')

            if np.abs(den_d) < 0.00001 or np.abs(num_d) < 0.00001:
                num_d = 0.0

            # if num_d < 0.000001 :
            #     num_d = 0
            # elif den_d < 0.000001 :
            #     den_d = 0

            a = round((num_d/den_d),2)

            # if (num_d / den_d) < 0.0001:
            #     a =0

            arr[j][k] = a
    arr1 = arr[:,1:]
    # print(arr1)
    ax = sns.heatmap(arr1,annot = True, xticklabels = np.arange(1,7))
    plt.title('Generalized partial autocorrelation function (GPAC)')
    plt.show()

