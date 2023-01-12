import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS
from itertools import product
import warnings
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as ETS
from statistics import median


## 資料整理
path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/final'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0.drop(y0.index[-1],inplace=True)
y = pd.Series(np.array(y0), index=pd.date_range("2003-01-01", periods=len(y0), freq="Q"), name="prunung").astype('float64')
y
yin = y.iloc[0:70,]
type(yin)
type(y)
len(yin)


##########################################
yin = pd.DataFrame(np.array(yin),pd.period_range(start="2003-1-1",periods=len(yin),freq="Q"))
## 找出最好的model
T = len(yin)
# model = AutoETS(auto=True,seasonal='add',sp=12)
error = ['add','mul']
trend = ['add','mul']
seasonal = ['add','mul']
damped_trend = [True, False]
models = list(product(error,trend,seasonal,damped_trend))
model = [None]*len(models)
y_ciu = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start='2003-1-1',periods=len(yin) ,freq='Q'))
yf = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start='2003-1-1',periods=len(yin) ,freq='Q'))
bic = np.zeros(len(models))+9999
for ii,jj in enumerate(models):
    try:
        model[ii] = AutoETS(sp=4,error=jj[0],trend=jj[1],seasonal=jj[2],damped_trend=jj[3]).fit(yin)
        yf.iloc[:,ii] = model[ii].predict(fh=pd.period_range(start='2003-01-01',periods=len(yin) ,freq='Q'))
        y_ciu.iloc[:,ii] = model[ii].predict_interval(fh=pd.period_range(start='2003-01-01',periods=len(yin) ,freq='Q')).iloc[:,1]
        bic[ii] = model[ii].get_fitted_params()['aicc']
    except ValueError:
        pass
bic

upCI = y_ciu.quantile(q=0.5,axis=1)+1.5*(y_ciu.quantile(q=0.75, axis=1)-y_ciu.quantile(q=0.25, axis=1))
lowerCI = y_ciu.quantile(q=0.5,axis=1)-1.5*(y_ciu.quantile(q=0.75, axis=1)-y_ciu.quantile(q=0.25,axis=1))
sel_index = list(range(len(models)))
for ii in range(len(models)):
    if np.sum(y_ciu.iloc[:,ii]<lowerCI)>0 or np.sum(y_ciu.iloc[:,ii]>upCI) >0:
        sel_index.remove(ii)
bic[sel_index]
best_model = model[sel_index[np.argmin(bic[sel_index])]]# np.argmin()，可以回傳最小值的index
best_model

## 方法3 : 用最佳的model預測
yin.shape
y.shape
yin0=pd.array(yin[0])
yin0.shape
yin = pd.Series(yin0, index=pd.date_range("2003-01-01", periods=len(yin), freq="Q"), name="prunung").astype('float64')
modelmmm = ETS(yin,error='mul',trend='mul',seasonal="mul", seasonal_periods=4).fit(disp=True)
y_pred = modelmmm.predict()
y_cast = modelmmm.forecast(9)
y_pred = np.array([y_pred]).T
y_cast = np.array([y_cast]).T
yAll = np.r_[y_pred, y_cast]
yPruning = pd.DataFrame(yAll,pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
yPruning
yPruning.index[-9]
## 做信賴區間
a=100
yci0 = modelmmm.simulate(anchor = "end", nsimulations=9, repetitions=a)
yci0
yci = pd.DataFrame(np.zeros((len(yPruning),a)),pd.period_range(start="2003-01-01",periods=len(yPruning),freq='Q'))
yci
for ii in range(a):
    yci.iloc[0:70,ii] = yPruning.iloc[0:70,0]
    yci.iloc[70:,ii] = yci0.iloc[0:,ii]
yci.plot(legend=False,color="grey",lw=0.1)
plt.show()


## MAPE
APE3 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - yPruning.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE3.append(per_err)
MAPE3 = sum(APE3)/len(APE3)
MAPE3
# >>> MAPE = 0.047383

## 畫圖
y = np.asarray(y)
y = pd.DataFrame(y,pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
yM8 = pd.concat([y,yPruning,yci],axis=1)
plotn = 0
yM8.iloc[:70,1] = yM8.iloc[:70,0]
for ii in range(2,a+1):
    yM8.iloc[:70,ii] = yM8.iloc[:70,0]
for ii in range(2,a+1):
    plotn = yM8.iloc[:,ii].plot(lw=0.5,color='grey',legend= False,alpha=0.5)
yM8.iloc[:,0].plot(lw=2,color='red',legend = False)
yM8.iloc[:,1].plot(lw=1,color='blue',legend  = False,alpha=0.5)
plt.show()



#################################################################








