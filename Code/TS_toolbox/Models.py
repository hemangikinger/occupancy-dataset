import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api  as sm
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.tsa.holtwinters as ets
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA

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

def roots(theta, na):
    x = theta
    den = x[:na] #err
    num = x[na:] #y

    if len(den) > len(num):
        diff = len(den)-len(num)
        num = np.pad(num, (0,diff), 'constant', constant_values=0)

    elif len(num) > len(den):
        diff = len(num)-len(den)
        den = np.pad(den, (0,diff), 'constant', constant_values=0)

    den = np.r_[1, den]
    num = np.r_[1, num]

    a=np.roots(num)
    b=np.roots(den)
    print('Poles and zeros')
    print(a)
    print(b)


def average_forecast_method(arr_train, arr_test):
    pred_train = []
    pred_test = []

    arr_train = np.array(arr_train)
    arr_test = np.array(arr_test)

    for i in range(1, len(arr_train) + 1):
        avg = arr_train[:(i)].sum() / (len(arr_train[:(i)]))
        pred_train.append(round(avg, 2))

    for j in range(len(arr_test)):
        pred_test.append(pred_train[-1])

    pred_train.pop()

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    plt.figure()
    plt.plot(arr_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in arr_train] + [x for x in arr_test], label='test set')
    plt.plot([None for i in arr_train] + [x for x in pred_test], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Average Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = arr_train[1:] - pred_train
    forecast_err = arr_test - pred_test

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by Average Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (Average Method) : ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for Average Method is {np.round(np.mean(residual_err), 2)}')

    print(f'MSE of residual error for Average Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for Average Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for Average Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (Average Method) : ', np.round(model_fit, 2))


def naive_method(arr_train,arr_test):
    pred_train = []
    pred_test = []

    for i in range(1, len(arr_train)):
        pred_train.append(arr_train[(i-1)])

    for j in range(len(arr_test)):
        # print(j)
        pred_test.append(arr_train[-1])

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    # print(pred_train)
    # print(pred_train)
    plt.figure()
    plt.plot(arr_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in arr_train] + [x for x in arr_test], label='test set')
    plt.plot([None for i in arr_train] + [x for x in pred_test], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Naive Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = arr_train[1:] - pred_train
    forecast_err = arr_test - pred_test

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by Naive Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (Naive Method): ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for Naive Method is {np.round(np.mean(residual_err), 2)}')

    print(f'MSE of residual error for Naive Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for Naive Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for Naive Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (Naive Method) : ', np.round(model_fit,2))

def drift_method(arr_train, arr_test):
    pred_train = []
    pred_test = []
    val1 = 0

    for i in range(2, len(arr_train)):
        val = arr_train[i - 1] + ((1) * ((arr_train[i - 1] - arr_train[0]) / (i - 1)))
        # val = arr_train[i] + (i + 1) * ((arr_train[i] - arr_train[0]) / i)
        pred_train.append(val)

    for j in range(len(arr_test)):
        val1 = arr_train[-1] + (j + 1) * ((arr_train[-1] - arr_train[0]) / (len(arr_train) - 1))
        pred_test.append(val1)

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    plt.figure()
    plt.plot(arr_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in arr_train] + [x for x in arr_test], label='test set')
    plt.plot([None for i in arr_train] + [x for x in pred_test], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Drift Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = arr_train[2:] - pred_train
    forecast_err = arr_test - pred_test

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by Drift Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (Drift Method): ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for Drift Method is {np.round(np.mean(residual_err), 2)}')

    print(f'MSE of residual error for Drift Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for Drift Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for Drift Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (Drift Method): ', np.round(model_fit,2))


def ses_method(arr_train,arr_test,alpha):

    pred_train = []
    pred_test = []
    val = 0
    val1 = 0

    for i in range(0, len(arr_train)):
        if i < 1:
            pred_train.append(arr_train[0])
        else:
            val = (alpha * arr_train[i-1] ) + ((1 - alpha)*pred_train[i-1])
            pred_train.append(val)

    for j in range(len(arr_test)):
        val1 = (alpha * arr_train[-1] ) + ((1 - alpha)*pred_train[-1])
        pred_test.append(val1)

    pred_train = np.array(pred_train)
    pred_test = np.array(pred_test)

    plt.figure()
    plt.plot(arr_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in arr_train] + [x for x in arr_test], label='test set')
    plt.plot([None for i in arr_train] + [x for x in pred_test], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - SES Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = arr_train - pred_train
    forecast_err = arr_test - pred_test

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by SES Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (SES Method) : ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for SES Method is {np.round(np.mean(residual_err), 2)}')

    print(f'MSE of residual error for SES Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for SES Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for SES Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (SES Method) : ', np.round(model_fit,2))


def holt_linear_method(y_train, y_test):

    holtt = ets.ExponentialSmoothing(y_train, trend='mul', damped_trend=False, seasonal=None).fit()
    pred_train_holtl = holtt.predict(start=0, end=(len(y_train) - 1))
    pred_test_holtl = holtt.forecast(steps=len(y_test))

    plt.figure()
    plt.plot(y_train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in y_train] + [x for x in y_test], label='test set')
    plt.plot([None for i in y_train] + [x for x in pred_test_holtl], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Holt-Linear Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = y_train - pred_train_holtl
    forecast_err = y_test - pred_test_holtl

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by Holt Linear Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (Holt-Linear Method): ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for Holt-Linear Method is {np.round(np.mean(residual_err), 2)}')

    print(f'MSE of residual error for Holt-Linear Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for Holt-Linear Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for Holt-Linear Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (Holt-Linear Method): ', np.round(model_fit,2))

def holt_winters_method(train,test):
    holtt1 = ets.ExponentialSmoothing(train, trend='mul', damped_trend=False, seasonal='mul',
                                      seasonal_periods=1440).fit()
    pred_train_holts = holtt1.predict(start=0, end=(len(train) - 1))
    pred_test_holts = holtt1.forecast(steps=len(test))

    plt.figure()
    plt.plot(train, label='training set', markerfacecolor='blue')
    plt.plot([None for i in train] + [x for x in test], label='test set')
    plt.plot([None for i in train] + [x for x in pred_test_holts], label='h-step forecast')
    plt.legend()
    plt.title('Temperature - Holt-Winter Seasonal Method & Forecast')
    plt.ylabel('Values')
    plt.xlabel('Number of Observations')
    plt.grid()
    plt.show()

    residual_err = train - pred_train_holts
    forecast_err = test - pred_test_holts

    val, lags, olags = autocorrelation(50, residual_err)

    plt.figure()
    plt.stem(lags, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(residual_err))), (1.96 / np.sqrt(len(residual_err))), alpha=0.2,
                color='blue')
    plt.title('Autocorrelation created from Residual Error by Holt-Winter Method ')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    Q = sm.stats.acorr_ljungbox(residual_err, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for training set (Holt-Winter Method): ", np.round(Q,2))

    model_fit = (np.var(residual_err) / np.var(forecast_err))

    print(f'Mean of residual error for Holt-Winter Method is {np.round(np.mean(residual_err), 2)}')

    print(f'Mean of residual error for Holt-Winter Method is {np.round(np.mean(residual_err ** 2), 2)}')

    print(f'Variance of residual error for Holt-Winter Method is {np.round(np.var(residual_err),2)}')

    print(f'Variance of forecast error for Holt-Winter Method is {np.round(np.var(forecast_err),2)}')

    print('variance of the residual errors versus the variance of the forecast errors (Holt-Winter Method): ', np.round(model_fit,2))


def ARIMA_method(na,nb, d, lags, y_train, y_test):

    model = ARIMA(y_train, order=(na, d, nb)).fit()

    print(model.summary())

    model_hat = model.predict()
    test = model.forecast(len(y_test))

    e = y_train - model_hat
    f = y_test - test
    val,lag,vals = autocorrelation(lags, e)

    squared_train_err = [number ** 2 for number in e]
    squared_test_err = [number ** 2 for number in f]


    print("\n")

    plt.figure()
    plt.stem(lag, val, markerfmt='C3o')
    plt.axhspan((-1.96 / np.sqrt(len(e))), (1.96 / np.sqrt(len(e))), alpha=0.2,
                color='blue')
    plt.title(f'ARIMA({na},{d},{nb}) Auto-Correlation Plot')
    plt.xlabel('# Lags')
    plt.ylabel('Correlation Value')
    plt.grid()
    plt.show()

    # Q = len(y_train) * np.sum(np.square(vals[lags:]))

    Q = sm.stats.acorr_ljungbox(e, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print("Q-Value for ARIMA residuals: ", Q)

    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)

    if Q < chi_critical:
        print("As Q-value is less than chi-2 critical, Residual is white")
    else:
        print("As Q-value is greater than chi-2 critical,Residual is NOT white")


    lst = ["{0:.2f}".format(np.sum(squared_test_err) / len(squared_test_err))
        , "{0:.2f}".format(np.sum(squared_train_err) / len(squared_train_err))
        , "{0:.2f}".format(np.var(e))
        , "{0:.2f}".format(np.var(f))
        , "{0:.2f}".format(Q)
        , "{0:.2f}".format(np.var(e) / np.var(f))]

    armais_df = pd.DataFrame(lst, columns=[f'ARIMA({na},{d},{nb})'],
                             index=['MSE_Fcast', 'MSE_Residual', 'Var_Pred', 'Var_Fcast', 'QValue_Residual',
                                    'Variance_Ratio'])



    model.plot_diagnostics(figsize=(14,10))
    plt.suptitle(f'ARIMA({na},{d},{nb}) Diagnostic Analysis')
    plt.grid()
    plt.show()

    # print(sm.stats.acorr_ljungbox(e, lags=[lags], boxpierce=True))

    for i in range(na):
        print('The AR coefficient a{}'.format(i), 'is:', (-1 * model.params[i]))
    for i in range(nb):
        print('The MA coefficient a{}'.format(i), 'is:', model.params[i + na ])

    for i in range(1, na + 1):
        print("The confidence interval for a{}".format(i), " is: ", -model.conf_int()[i][0], " and ",
              -model.conf_int()[i][1])

    for i in range(1, nb + 1):
        print("The confidence interval for b{}".format(i), " is: ", model.conf_int()[i + na][0], " and ",
              model.conf_int()[i + na][1])

    print("\n")

    plt.figure()
    plt.plot(y_train, 'r', label='Train data')
    plt.plot(model_hat, 'b', label='Fitted data')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(f'Stats ARIMA ({na},{d},{nb}) model and predictions')
    plt.grid()
    plt.show()


    # print(model.params)
    roots(model.params[:(na+nb)],na)
    # fc_series = pd.Series(fc, index=test.index)

    print("\n")

    print(armais_df)

    plt.figure()
    plt.plot(y_test, 'r', label='Test data')
    plt.plot(test, 'b', label='Forecasted data')
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title(f'Stats ARIMA ({na},{d},{nb}) model and Forecast')
    plt.grid()
    plt.show()

