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

## 資料整理

path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/final'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0.drop(y0.index[-1],inplace=True)
y = pd.Series(np.array(y0), index=pd.date_range("2003-01-01", periods=len(y0), freq="Q"), name="TheManufacturingIndustry")
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


## #MBB + Forcast, AutoETS預測，預測9期
yin = y.iloc[0:70,] #trainingset
#STL
stl = STL(yin,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
# res.plot()
# plt.show()
# dir(res)
l = 8
B = 101
T1 = len(y)
fcast_h = list(range(1,10))#預測9期
bt_yin = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
y_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start="2003-01-01",periods=T1,freq="Q"))
for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal
    y_fcast.iloc[len(yin):,bb] = AutoETS(auto=True,n_jobs=-1,sp=4,maxiter=5000).fit_predict(bt_yin.iloc[:,bb],fh=fcast_h)
    # fh: forecasting horizon 預測範圍
    y_fcast.iloc[:len(yin),bb] = yin


# bt_yin.plot(legend=False,lw=0.1,color="grey")
# y.plot(color="red",lw=0.2)
# plt.show()

####################################################################


# 方法_課本：找到中位數的預測數列
sumPreMedian  = pd.DataFrame(np.zeros((1,B)))
for ii in range(B):
    sumPreMedian[ii] = y_fcast.iloc[:,ii].sum()
median_value = np.median(sumPreMedian)
median_index = np.where(sumPreMedian == median_value)[1][0]
medianclass_fcast = y_fcast[median_index]
len(medianclass_fcast)
type(medianclass_fcast)
medianclass_fcast.shape
medianclass_fcast = pd.Series(np.array(medianclass_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" ),name="Mclass").astype("float64")
y_fcast1 =  pd.DataFrame(np.array(y_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" )).astype("float64")

#MAPE
APE1 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - medianclass_fcast.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE1.append(per_err)
MAPE1 = sum(APE1)/len(APE1)
MAPE1
# >>> MAPE1 real    0.12118448113406559  0.11434579881274463  0.12109562269237611 0.12222636809248982

####################################################################


# 方法二：重組各條預測數列的中位數
median_fcast = np.median(y_fcast,axis=1)
# median_fcast
# median_fcast.shape
# type(median_fcast)
median_fcast = pd.Series(np.array(median_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" ),name="M1").astype("float64")
y_fcast2 =  pd.DataFrame(np.array(y_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" )).astype("float64")

# MAPE
APE2 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - median_fcast.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE2.append(per_err)
MAPE2 = sum(APE2)/len(APE2)
MAPE2
# >>> MAPE2 0.12080688313472424 0.11835559236766498  0.12181814718533289 0.12355871771234035
####################################################################


## 方法三 :pruning
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
yPruning.index[-9]
## 做信賴區間
a=101
yci0 = modelmmm.simulate(anchor = "end", nsimulations=9, repetitions=a)
yciP = pd.DataFrame(np.zeros((len(yPruning),a)),pd.period_range(start="2003-01-01",periods=len(yPruning),freq='Q'))
for ii in range(a):
    yciP.iloc[0:70,ii] = yPruning.iloc[0:70,0]
    yciP.iloc[70:,ii] = yci0.iloc[0:,ii]
# yci.plot(legend=False,color="grey",lw=0.1)
# plt.show()


## MAPE
APE3 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - yPruning.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE3.append(per_err)
MAPE3 = sum(APE3)/len(APE3)
MAPE3
# >>> MAPE3 0    0.047383
####################################################################################


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
yciPB_fcast = yPruning_fcast
for ii in range(B):
    yciPB_fcast.iloc[0:70,ii] = yin.iloc[0:70]
# yciPB_fcast.plot(legend = False,color="grey",lw=0.1)
# plt.show()

## MAPE
APE4 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - medianPruning1_fcast.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE4.append(per_err)
MAPE4 = sum(APE4)/len(APE4)
MAPE4
# >>> MAPE4 0.13970989046659904
####################################################################################


##畫圖
y = pd.DataFrame(np.asarray(y),pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
y = y.rename(columns={0:'real'})
for ii in range(0,101):
    yciP.iloc[0:70,ii] = y.iloc[0:70,0]
for ii in range(0,101):
    y_fcast1.iloc[0:70,ii] = y.iloc[0:70,0]
yciPB_fcast = pd.DataFrame(np.asarray(yciPB_fcast),pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
#第一種圖：比較四個預測結果
yM10 = pd.concat([y,medianclass_fcast,median_fcast,yPruning,medianPruning1_fcast],axis=1)
yM10 = yM10.rename(columns={"Mclass":'Bagged.BLD.MBB.ETS','M1':"My Method",0:"Pruned ETS",'MPruningclass':'Pruned Bagged ETS'})
for ii in range(1,5):
    yM10.iloc[0:70,ii] = yM10.iloc[0:70,0]
yM10.iloc[:,0].plot(lw=1.5,color='black',legend=True)
yM10.iloc[70:,1].plot(lw=1,color='blue',alpha=0.5,legend=True)
yM10.iloc[70:,2].plot(lw=1,color='green',alpha=0.5,legend=True)
yM10.iloc[70:,3].plot(lw=1,color='red',alpha=1,legend=True)
yM10.iloc[70:,4].plot(lw=1,color='orange',alpha=0.5,legend=True)
plt.show()
#第二種圖：比較四個預測結果，並把CI畫進去
yci10 = pd.concat([yM10,y_fcast1,yciP,yciPB_fcast],axis=1)
for ii in range(1,5):
    yci10.iloc[0:70,ii] = yci10.iloc[0:70,0]
yci10.iloc[:,0].plot(lw=1.5,color='black',legend=True,figsize=[12,8])
yci10.iloc[70:,1].plot(lw=1,color='blue',legend=True,figsize=[12,8])
yci10.iloc[70:,2].plot(lw=1,color='green',legend=True,figsize=[12,8])
yci10.iloc[70:,3].plot(lw=1,color='red',legend=True,figsize=[12,8])
yci10.iloc[70:,4].plot(lw=1,color='orange',legend=True,figsize=[12,8])
plotn1 = 0
for ii in range(5,106):
    plotn1 = yci10.iloc[:,ii].plot(lw=0.5,color='blue',legend= False,alpha=0.1,figsize=[12,8])
plotn2 = 0
for ii in range(107,207):
    plotn2 = yci10.iloc[:,ii].plot(lw=0.5,color='red',legend= False,alpha=0.1,figsize=[12,8])
plotn3 = 0
for ii in range(208,308):
    plotn3 = yci10.iloc[:,ii].plot(lw=0.5,color='orange',legend= False,alpha=0.1,figsize=[12,8])
plt.show()


########################################################################################



y_fcast
yciP
yciPB_fcast
yciPB_fcast = pd.DataFrame(np.asarray(yciPB_fcast),pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
for ii in range(1,5):
    yciP.iloc[0:70,ii] = yM10.iloc[0:70,0]
for ii in range(1,5):
    y_fcast.iloc[0:70,ii] = yM10.iloc[0:70,0]
yci10 = pd.concat([y,medianclass_fcast,median_fcast,yPruning,medianPruning1_fcast,y_fcast,yciP,yciPB_fcast],axis=1)
