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
path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/期末報告'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0.drop(y0.index[-1],inplace=True)
y = pd.Series(np.array(y0), index=pd.date_range("2003-01-01", periods=len(y0), freq="Q"), name="TheManufacturingIndustry")

## Define a function for moving block bootstrap
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

## 分解TS (STL)
yin = y.iloc[0:70,]
stl = STL(yin,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal

## MBB 創造1000條TS
l = 8
B = 1001
bt_yin = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal

## 找到Median
bt_yin
sumPreMedian  = pd.DataFrame(np.zeros((1,B)))
for ii in range(B):
    sumPreMedian[ii] = bt_yin.iloc[:,ii].sum()
median_value = np.median(sumPreMedian)
median_index = np.where(sumPreMedian == median_value)
median_TS = bt_yin[median_index[1]]
median_TS = np.array(median_TS)
type(median_TS)
median_TS.shape

median_ts = np.zeros((70))
for ii in range(len(median_TS)):
    median_ts[ii] = median_TS[ii,0]
median_TS = median_ts
median_TS = pd.Series(np.array(median_TS),index=pd.period_range(start="2003-1-1" ,periods=len(yin) ,freq="Q" ),name="TheManufacturingIndustry3").astype("float64")
##########################################

## 找出最好的model
T = len(median_TS)
model = AutoETS(auto=True,seasonal='add',sp=4)
error=["add","mul"]
trend=["add",'mul']
seasonal=['add','mul']
damped_trend = [True, False]#未知
models = list(product(error,trend,seasonal,damped_trend))
model = [None]*len(models)
y_ciu = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start="2003-1-1",periods=len(yin),freq="Q"))
yf = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start='2003-1-1',periods=len(yin),freq='Q'))
bic = np.zeros(len(models))+9999
for ii,jj in enumerate(models):
    try:
        model[ii] = AutoETS(sp=4, error=jj[0],trend=jj[1],seasonal=jj[2],damped_trend=jj[3]).fit(median_TS)
        bic[ii] = model[ii].get_fitted_params()['aicc']
        yf.iloc[:,ii] = model[ii].predict(fh=pd.period_range(start='2003-1-1',periods=len(y),freq='Q'))
        y_ciu.iloc[:,ii] = model[ii].predict_interval(fh=pd.period_range('2003-1-1',periods=len(yin),freq="Q")).iloc[:,1]
    except ValueError:
        pass

upCI = y_ciu.quantile(q=0.5,axis=1)+1.5*(y_ciu.quantile(q=0.75, axis=1)-y_ciu.quantile(q=0.25, axis=1))
lowerCI = y_ciu.quantile(q=0.5,axis=1)-1.5*(y_ciu.quantile(q=0.75, axis=1)-y_ciu.quantile(q=0.25,axis=1))
sel_index = list(range(len(models)))
for ii in range(len(models)):
    if np.sum(y_ciu.iloc[:,ii]<lowerCI)>0 or np.sum(y_ciu.iloc[:,ii]>upCI) >0:
        sel_index.remove(ii)
bic[sel_index]
best_model = model[sel_index[np.argmin(bic[sel_index])]]# np.argmin()，可以回傳最小值的index
best_model

## 用最佳的model預測
modelmmm = ETS(median_TS,error='mul',trend='mul',seasonal="mul", seasonal_periods=4).fit(disp=True)
y_pred = modelmmm.predict()
y_cast = modelmmm.forecast(9)
y_pred = np.array([y_pred]).T
y_cast = np.array([y_cast]).T
yAll = np.r_[y_pred, y_cast]
yAll = pd.DataFrame(yAll,pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))

## 做信賴區間
a=100
yci = modelmmm.simulate(anchor = "end", nsimulations=9, repetitions=a)

## 畫圖
y = np.asarray(y)
y = pd.DataFrame(y,pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
yCi = pd.concat([y,yci],axis=1)
for ii in range(len(y)-len(yci)):
    yCi.iloc[ii,1:] = yCi.iloc[ii,1]
# 繪製具有不同線寬的資料
for ii in range(2,a):
    plotn = yCi.iloc[:,ii].plot(lw=0.5,color='grey',legend= False,alpha=0.5)
plotn.plot()
yCi.iloc[:,0].plot(lw=2,color='red',legend = False)
plt.show()








#可以指定日期 跟 合併兩個資料的方法
# ax = y_ciu['2003':'2022'].plot(legend=False,color='gray',linewidth=0.1)
# pd.concat((upCI,lowerCI),axis=1)['2003':'2022'].plot(ax=ax,color='red',legend=False,lw=0.2)
# 代表使用橫軸在做合併dataframe，也就是有很多column)，若是用縱軸axis=0，也就是有很多row
# plt.show()

#印出全部日期 跟 合併兩個資料的方法
# ax = y_ciu.plot(legend=False,color='gray',linewidth=0.1)
# pd.concat((upCI,lowerCI),axis=1).plot(ax=ax,color='red',legend=False,lw=0.2)
# plt.show()

#拆分的方法
# y_ciu.plot(color="grey",legend=False,linewidth=0.1)
# upCI.plot(color='red')
# lowerCI.plot(color='blue')
# plt.show()