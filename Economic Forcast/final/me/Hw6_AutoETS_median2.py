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

# 方法二：重組各條預測數列的中位數
median_fcast = np.median(y_fcast,axis=1)
# median_fcast
# median_fcast.shape
# type(median_fcast)
median_fcast = pd.Series(np.array(median_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" ),name="M1").astype("float64")
y_fcast =  pd.DataFrame(np.array(y_fcast),index=pd.period_range(start="2003-1-1" ,periods=len(y) ,freq="Q" )).astype("float64")

# MAPE
APE2 = []
for day in range(70,79):
    per_err = (y.iloc[day,] - median_fcast.iloc[day]) / y.iloc[day]
    per_err = abs(per_err)
    APE2.append(per_err)
MAPE2 = sum(APE2)/len(APE2)
MAPE2
# >>> MAPE2 0.122527

##畫圖
y = np.asarray(y)
# np.shape(y)
y = pd.DataFrame(y,pd.period_range(start="2003-1-1",periods=len(y),freq="Q"))
y = y.rename(columns={0:'real'})
yM4 = pd.concat([y,median_fcast,y_fcast],axis=1)
plotn = 0
for ii in range(2,B+2):
    plotn = yM4.iloc[:,ii].plot(lw=0.5,color='grey',legend= False,alpha=0.5)
yM4.iloc[:,0].plot(lw=2,color='red',legend = False)
yM4.iloc[:,1].plot(lw=1,color='blue',legend  = False,alpha=0.5)
plt.show()

########################################################################################






#len(y_fcast)
# y_fcast.loc["2003-03-01":].plot(legend=False,figsize=[12,4],lw=1)
# y.loc["2003-03-01":].plot(legend=False,figsize=[12,4],lw=2,color="red")
# plt.show()