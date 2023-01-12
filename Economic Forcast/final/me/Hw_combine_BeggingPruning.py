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

# 資料整理

path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/期末報告'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0.drop(y0.index[-1],inplace=True)
y = pd.Series(np.array(y0), index=pd.date_range("2003-01-01", periods=len(y0), freq="Q"), name="TheManufacturingIndustry")

# define a function for moving block bootstrap
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


########################################################################################
# 這邊用AutoETS預測，預測9期，沒有用median的做法
yin = y.iloc[0:70,]
#STL
stl = STL(yin,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
l = 8
B = 10
T1 = len(y)
fcast_h = list(range(1,10)) 
#預測9期
bt_yin = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
y_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start="2003-01-01",periods=T1,freq="Q"))

for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal
    y_fcast.iloc[len(yin):,bb] = AutoETS(auto=True,n_jobs=-1,sp=4,maxiter=5000).fit_predict(bt_yin.iloc[:,bb],fh=fcast_h)
    y_fcast.iloc[:len(yin),bb] = yin

# y_fcast.loc["2003-03-01":].plot(legend=False,figsize=[15,5],lw=1)
# y.loc["2003-03-01":].plot(legend=False,figsize=[15,5],lw=2,color="red")
# plt.show()
########################################################################################




# 想測試如何找到midian的series

#STL
yin = y.iloc[0:70,]
stl = STL(yin,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
# MBB 創造100條TS
l = 8
B = 5
bt_y = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal

# bt_y.loc["2003-03-01":].plot(legend=False,figsize=[15,5],lw=1)
# plt.show()

# 找到median 的 Time Series，這邊的median是在重組剛剛所以有TS的element中的中位數
# 所以新得到的TS是從好幾個TS中獲得的
median_TS = np.median(bt_yin,axis=1)
len(median_TS)
median_TS = np.asarray(median_TS)
# 幫Time Series編號
type(median_TS)
median_TS = pd.Series(np.array(median_TS),index=pd.period_range(start="2003-1-1" ,periods=len(yin) ,freq="Q" ),name="TheManufacturingIndustry2").astype("float64")
# 找出最好的model
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

#可以指定日期 跟 合併兩個資料的方法
# ax = y_ciu['2003':'2022'].plot(legend=False,color='gray',linewidth=0.1)
# pd.concat((upCI,lowerCI),axis=1)['2003':'2022'].plot(ax=ax,color='red',legend=False,lw=0.2)
# 代表使用橫軸在做合併dataframe，也就是有很多column)，若是用縱軸axis=0，也就是有很多row
# plt.show()

#印出全部日期 跟 合併兩個資料的方法
# ax = y_ciu.plot(legend=False,color='gray',linewidth=0.1)
# pd.concat((upCI,lowerCI),axis=1).plot(ax=ax,color='red',legend=False,lw=0.2)
# plt.show()

# 拆分的方法
y_ciu.plot(color="grey",legend=False,linewidth=0.1)
upCI.plot(color='red')
lowerCI.plot(color='blue')
plt.show()


sel_index = list(range(len(models)))
for ii in range(len(models)):
    if np.sum(y_ciu.iloc[:,ii]<lowerCI)>0 or np.sum(y_ciu.iloc[:,ii]>upCI) >0:
        sel_index.remove(ii)
bic[sel_index]
best_model = model[sel_index[np.argmin(bic[sel_index])]]
# np.argmin()，可以回傳最小值的index
# 方法一：用最佳的model預測
a= best_model
modelmmm = ETS(median_TS,error='mul',trend='mul',seasonal="mul", seasonal_periods=4).fit(disp=True)
y_pred = modelmmm.predict()
y_cast = modelmmm.forecast(9)
y_pred = np.array([y_pred]).T
y_cast = np.array([y_cast]).T
yAll = np.r_[y_pred, y_cast]
yAll = pd.DataFrame(yAll,pd.period_range(start="2003-1-1",periods=len(y),freq="Q",name="MBest"))

# 方法二：用加權的model預測
weights = np.exp(-0.5*(bic[sel_index]-np.min(bic[sel_index])))/np.sum(np.exp(-0.5*(bic[sel_index]-np.min(bic[sel_index]))))
y_ave = yf[sel_index]@weights
len(y_ave)


y=np.asarray(y)
y = pd.DataFrame(y,pd.period_range(start="2003-1-1",periods=len(y),freq="Q",name="real"))



yallll= pd.concat([y,yAll,y_ave],axis=1)
yallll.plot()
plt.show()