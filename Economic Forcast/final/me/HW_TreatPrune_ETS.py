import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = '/Users/liurenhao/Documents/TaipeiuniversityNote/EcnomicForcast/期末報告'
dat = pd.read_excel(path+'/TheManufacturingIndustry .xlsx', sheet_name='Report')
y0 = dat["電腦、電子產品及光學製品製造業"]
y0 = dat.iloc[1:,1]
y0
y0.drop(y0.index[-1],inplace=True)
y = pd.Series(np.array(y0),index=pd.period_range(start="2003-1-1" ,periods=len(y0) ,freq="Q" ),name="TheManufacturingIndustry").astype("float64")
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sktime.forecasting.ets import AutoETS
from itertools import product
import warnings
T = len(y)
model = AutoETS(auto=True,seasonal='add',sp=4)

error=["add","mul"]
trend=["add",'mul']
seasonal=['add','mul']
damped_trend = [True, False]
#未知
models = list(product(error,trend,seasonal,damped_trend))
# product()會產出一堆組合
#[('add', 'add', 'add', True), 
# ('add', 'add', 'add', False), 
# ('add', 'add', 'mul', True), 
# ('add', 'add', 'mul', False), 
# ('add', 'mul', 'add', True), 以此類推
# 把這些組合用成list
model = [None]*len(models)
#最終會變成長度為16的list，裡面的內容物會填滿各種model組合
y_ciu = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start="2003-1-1",periods=len(y0),freq="Q"))
yf = pd.DataFrame(np.zeros((T,len(models))),pd.period_range(start='2003-1-1',periods=len(y0),freq='Q'))
bic = np.zeros(len(models))+9999
for ii,jj in enumerate(models):
    try:
        model[ii] = AutoETS(sp=4, error=jj[0],trend=jj[1],seasonal=jj[2],damped_trend=jj[3]).fit(y)
        model[ii]
        yf.iloc[:,ii] = model[ii].predict(fh=pd.period_range(start='2003-1-1',periods=len(y0),freq='Q'))
        y_ciu.iloc[:,ii] = model[ii].predict_interval(fh=pd.period_range('2003-1-1',periods=len(y0),freq="Q")).iloc[:,1]
        bic[ii] = model[ii].get_fitted_params()['aicc']
    except ValueError:
        pass

bic
# models
# len(models)
# model[ii]
# model[ii].predict_interval(fh=pd.period_range(start='2003-1-1',periods=len(y0) ,freq='Q')).iloc[:,0]
# model[ii].predict(fh=pd.period_range(start='2003-1-1',periods=len(y0) ,freq='Q'))

upCI = y_ciu.quantile(q=0.5,axis=1)+1.5*(y_ciu.quantile(q=0.75,axis=1)-y_ciu.quantile(q=0.25,axis=1))
lowerCI = y_ciu.quantile(q=0.5, axis=1)-1.5*(y_ciu.quantile(q=0.75, axis=1)-y_ciu.quantile(q=0.25, axis=1))
ax = y_ciu["2003":'2022'].plot(legend=False,color='gray',lw=0.1)
pd.concat((upCI,lowerCI),axis=1)['2003':'2022'].plot(ax=ax,color='red',legend=False,lw=0.7)
plt.show()

sel_index = list(range(len(models))) 
# list中有16個空殼
for ii in range(len(models)):
    if np.sum(y_ciu.iloc[:,ii]<lowerCI)>0 or np.sum(y_ciu.iloc[:,ii]>upCI)>0:
        sel_index.remove(ii)
sel_index
best_model = model[sel_index[np.argmin(bic[sel_index])]]
sel_index#剛篩出來沒有outlier的model
bic[sel_index]#找出個個model的bic
np.argmin(bic[sel_index])#選出最小bic的model
model[sel_index[np.argmin(bic[sel_index])]]#獲得最小的model
weights = np.exp(-0.5*(bic[sel_index]-np.min(bic[sel_index])))/np.sum(np.exp(-0.5*(bic[sel_index]-np.min(bic[sel_index]))))
#paper權重的公式
y_ave = yf[sel_index]@weights
sel_index
yf[sel_index]
# 選出剛剛沒有outlier的model所預測的y
# @:有乘上去相加的感覺
y_ave
pd.concat((y,y_ave),axis=1).plot()
plt.show()

len(y0)