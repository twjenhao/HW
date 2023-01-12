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
fcast_h = list(range(1,10))#預測九期
len(fcast_h) 
#預測9期
bt_yin = pd.DataFrame(np.zeros((len(yin),B)),index=yin.index)
y_fcast=pd.DataFrame(np.zeros((T1,B)),index=pd.date_range(start="2003-01-01",periods=T1,freq="Q"))
len(y_fcast)
for bb in range(B):
    z = mbb(res.resid,l)
    bt_yin.iloc[:,bb] = np.array(z)+trend+seasonal
    y_fcast.iloc[len(yin):,bb] = AutoETS(auto=True,n_jobs=-1,sp=4,maxiter=5000).fit_predict(bt_yin.iloc[:,bb],fh=fcast_h)
    # fh: forecasting horizon 預測範圍
    y_fcast.iloc[:len(yin),bb] = yin


median_value = np.median(y_fcast.iloc[:,])






y_fcast.plot(legend=False,figsize=[12,4],lw=1,color="grey")
y.plot(legend=False,figsize=[12,4],lw=2,color="red")
plt.show()

########################################################################################

len(y_fcast)

# y_fcast.loc["2003-03-01":].plot(legend=False,figsize=[12,4],lw=1)
# y.loc["2003-03-01":].plot(legend=False,figsize=[12,4],lw=2,color="red")
# plt.show()