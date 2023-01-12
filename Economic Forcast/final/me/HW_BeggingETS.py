import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/期末報告'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0.drop(y0.index[-1],inplace=True)
y0
y = pd.Series(np.array(y0), index=pd.date_range("1-1-2003", periods=len(y0), freq="Q"), name="TheManufacturingIndustry")
y
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS
from itertools import product
import warnings

stl = STL(y,seasonal=5)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
res.plot()
plt.show()
len(y)

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

l = 8
B = 100
bt_y = pd.DataFrame(np.zeros((len(y),B)),index=y.index)
for bb in range(B):
    z = mbb(res.resid,l)
    bt_y.iloc[:,bb]= np.array(z)+trend+seasonal


bt_y.plot(legend=False)
plt.show()









idx = np.random.randint(n-l,size=nb)
idx
array([38, 20, 67, 14, 32, 57, 58, 70, 66, 23, 37])
n=len(y)
z=[]
z1 = z+list(res.resid[62:62+8])
z1
z2 = z1+list(res.resid[58:58+8])
z2
z3 = z2+list(res.resid[10:10+8])
z4 = z3+list(res.resid[40:40+8])
z5 = z4+list(res.resid[61:61+8])
z6 = z5+list(res.resid[49:49+8])
z7 = z6+list(res.resid[59:59+8])
z8 = z7+list(res.resid[8:8+8])
z9 = z8+list(res.resid[69:69+8])
z10 = z9+list(res.resid[18:18+8])
z11 = z10+list(res.resid[68:68+8])
len(z11)
len(z11)
z12 = z11[np.random.randint(l):]
len(z12)
z13 = z12[:n]
len(z13)
