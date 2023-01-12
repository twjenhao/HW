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

## define a function for moving block bootstrap
def mbb(x,l):
    n = len(x) 
    nb = np.int32(n/l)+2 
    idx = np.random.randint(n-l,size=nb)
    z = []
    for ii in idx:
        z = z+list(x[ii:ii+l]) 
    z = z[np.random.randint(l):]
    z = z[:n]
    return(z)


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
sel_index
bic[sel_index]
best_model = model[sel_index[np.argmin(bic[sel_index])]]# np.argmin()，可以回傳最小值的index
best_model


## 方法四 :pruning begging
yin = y.iloc[0:70,] #trainingset
stl = STL(yin,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
l = 8
B = 101
T1 = len(y)
fcast_h = list(range(1,10))#預測9期
bt_yin = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
yPruning_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start="2003-01-01",periods=T1,freq="Q"))
for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal
    modelmmm = ETS(bt_yin.iloc[:,bb],error='mul',trend='mul',seasonal="mul", seasonal_periods=4).fit()
    y_pred = modelmmm.predict()
    y_cast = modelmmm.forecast(9)
    y_pred = np.array([y_pred]).T
    y_cast = np.array([y_cast]).T
    yAll = np.r_[y_pred, y_cast]
    yAll = pd.DataFrame(np.array(yAll),index=pd.date_range(start="2003-01-01",periods=T1,freq="Q"))
    yPruning_fcast.iloc[:,bb] = yAll
sumPreMedianPruning1  = pd.DataFrame(np.zeros((1,B)))
for ii in range(B):
    sumPreMedianPruning1[ii] = yPruning_fcast.iloc[:,ii].sum()
median_value_Pruning1 = np.median(sumPreMedianPruning1)
median_index_Pruning1 = np.where(sumPreMedianPruning1 == median_value_Pruning1)[1][0]
medianPruning1_fcast = yPruning_fcast[median_index_Pruning1]
len(medianPruning1_fcast)
type(medianPruning1_fcast)
medianPruning1_fcast.shape
medianPruning1_fcast = pd.Series(np.array(medianPruning1_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" ),name="MPruningclass").astype("float64")


## 做信賴區間
yciPruning_fcast = yPruning_fcast
for ii in range(B):
    yciPruning_fcast.iloc[0:70,ii] = yin.iloc[0:70]
yciPruning_fcast.plot(legend = False,color="grey",lw=0.1)
plt.show()

## MAPE

APE4 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - medianPruning1_fcast.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE4.append(per_err)
MAPE4 = sum(APE4)/len(APE4)
MAPE4


## 畫圖

y = pd.DataFrame(np.asarray(y),pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
yci9 = pd.DataFrame(np.array(yci9),pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
yPBM9 = pd.concat([y,medianPruning1_fcast,yci9],axis=1)
plotn = 0
for ii in range(2,B+1):
    plotn = yPBM9.iloc[:,ii].plot(lw=0.5,color='grey',legend= False,alpha=0.5)
yPBM9.iloc[:,0].plot(lw=2,color='red',legend = False)
yPBM9.iloc[:,1].plot(lw=1,color='blue',legend  = False,alpha=0.5)
plt.show()



