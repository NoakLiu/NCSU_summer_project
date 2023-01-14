#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dftsla=pd.read_csv(".\TSLA.csv")
dffb=pd.read_csv(".\FB.csv")


# In[4]:


dftsla["maxDelta"]=dftsla["High"]-dftsla["Low"]
dftsla["Delta"]=abs(dftsla["Close"]-dftsla["High"])
dftsla['changePrecentage'] = dftsla['Close'] / dftsla['Open'] - 1


# In[5]:


#columns:--Date Open High Low Close Adj_Close Volume
print(dftsla.info())
print(dftsla.columns)
print(dftsla.head())
print(dftsla.tail())
print(dftsla.describe())
print(dftsla.describe().T)
print(dftsla['Date'].describe())
print(dftsla['Open'].mean())
print(dftsla['High'].unique())
print(dftsla['Low'].unique())
print(dftsla['Close'].unique())
print(dftsla.shape)


# In[6]:


dffb["maxDelta"]=dffb["High"]-dffb["Low"]
dffb["Delta"]=abs(dffb["Close"]-dffb["High"])


# In[7]:


#columns:--Date Open High Low Close Adj_Close Volume
print(dffb.info())
print(dffb.columns)
print(dffb.head())
print(dffb.tail())
print(dffb.describe())
print(dffb.describe().T)
print(dffb['Date'].describe())
print(dffb['Open'].mean())
print(dffb['High'].unique())
print(dffb['Low'].unique())
print(dffb['Close'].unique())
print(dffb.shape)


# In[8]:


print(dftsla.corr())


# In[9]:


ax=sns.heatmap(dftsla.corr())#data of heatmap must be numeric matrix


# In[10]:


print(dffb.corr())


# In[11]:


ax=sns.heatmap(dffb.corr())#data of heatmap must be numeric matrix


# In[66]:


#用热图呈现相关性

myCols=['Open', 'Close', 'High','Low']
ax=sns.heatmap(dffb[myCols].corr())
plt.title('Correlation')
plt.show()


# In[67]:


sns.distplot(dffb['Open'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('Open')
plt.show()
sns.distplot(dffb['Close'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('Close')
plt.show()


# In[69]:


x = dffb['Date']
y1 = dffb['High']
y2 = dffb['Low']
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1,'r')
ax1.set_ylabel('High Price')
ax1.set_title("High & Low #FB")
ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, 'g')
ax2.set_ylabel('Low Price')
ax2.set_xlabel('Date')
plt.show()

x = dffb['Date']
y1 = dffb['Open']
y2 = dffb['High']
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(x, y1,color='brown')
ax1.set_ylabel('Open Price')
ax1.set_title("Open & High #FB")
ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2, color='pink')
ax2.set_ylabel('High Price')
ax2.set_xlabel('Date')
plt.show()


# In[64]:


#Define changePrecentage(column):
dftsla['changePercentage'] = dftsla['Close'] / dftsla['Open'] - dftsla['Open'] / dftsla['Open']
print(dftsla['changePercentage'])
dffb['changePercentage'] = dffb['Close'] / dffb['Open'] - dffb['Open'] / dffb['Open']
print(dffb['changePercentage'])

#Draw Histogram
sns.distplot(dftsla['changePercentage'], hist=True, kde=True,
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(dffb['changePercentage'], hist=True, kde=True,
             bins=int(180/5), color = 'orange',
             hist_kws={'edgecolor':'brown'},
             kde_kws={'linewidth': 4})

plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data1 = np.array(dffb['Close'])
data2 = np.array(dftsla['Close'])


def do_linearRegression(data,name):
    y = data[0:1114] #to 2019/12/31
    X = np.linspace(0, 1114, 1114).reshape(-1, 1)
    X_future = np.linspace(1115, 1510, 1510-1115+1).reshape(-1, 1)
    y_future = data[1115:]
    model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    model.fit(X, y)

    predicted = model.predict(X)
    pred = model.predict(X_future)

    mse = np.sum(np.square(abs(y - predicted))) / 1000
    print(mse)

    plt.figure()
    plt.title(name)
    plt.plot(X, y, c='b')#real
    plt.plot(X, predicted, c='r')#regression output
    plt.plot(X_future, pred, c='green')#predict
    plt.plot(X_future, y_future, c='purple')#real
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()


# In[13]:


do_linearRegression(data1,'FB')


# In[14]:


do_linearRegression(data2,'TSLA')


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

datad1 = [np.array(dffb['Delta']),np.array(dffb['maxDelta'])]
datad2 = [np.array(dftsla['Delta']),np.array(dftsla['maxDelta'])]


def do_linearRegression2(data,name):
    y = data[0][0:1001] 
    X = data[1][0:1001].reshape(-1, 1)
    X_future = data[1][1001:1101].reshape(-1, 1)
    y_future = data[1][1001:1101]
    model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    model.fit(X, y)

    predicted = model.predict(X)
    pred = model.predict(X_future)

    mse = np.sum(np.square(abs(y - predicted))) / 1000
    print(mse)

    plt.figure()
    plt.title(name)
    plt.scatter(X, y, c='b')#real
    plt.plot(X, predicted, c='r')#regression output
    plt.plot(X_future, pred, c='green')#predict
    plt.scatter(X_future, y_future, c='purple')#real
    plt.xlabel("maxDelta")
    plt.ylabel("Delta")
    plt.show()


# In[16]:


do_linearRegression2(datad1,"FB")


# In[17]:


do_linearRegression2(datad2,"TSLA")


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

datapv1 = [np.array(dffb['Open']),np.array(dffb['Volume'])]
datapv2 = [np.array(dftsla['Open']),np.array(dftsla['Volume'])]


def do_linearRegression3(data,name):
    y = data[0][0:1001] 
    X = data[1][0:1001].reshape(-1, 1)
    X_future = data[1][1001:1101].reshape(-1, 1)
    y_future = data[1][1001:1101]
    model = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    model.fit(X, y)

    predicted = model.predict(X)
    pred = model.predict(X_future)

    mse = np.sum(np.square(abs(y - predicted))) / 1000
    print(mse)

    plt.figure()
    plt.title(name)
    plt.scatter(X, y, c='b')#real
    plt.plot(X, predicted, c='r')#regression output
    plt.plot(X_future, pred, c='green')#predict
    #plt.scatter(X_future, y_future, c='purple')#real
    plt.xlabel("Price")
    plt.ylabel("Volume")
    plt.show()


# In[19]:


do_linearRegression3(datapv1,"FB")


# In[20]:


do_linearRegression3(datapv2,"TSLA")


# In[21]:


#Lasso and Elastic Net for Sparse Signals for Two Companies

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# #############################################################################
# Generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 100
X = np.random.randn(n_samples, n_features)

# Decreasing coef w. alternated signs for visualization
idx = np.arange(n_features)
coef = (-1) ** idx * np.exp(-idx / 10)
coef[10:] = 0  # sparsify coef
y = np.dot(X, coef)

# Add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
ddata1 = [np.array(dffb['Close']),"FB"]
ddata2 = [np.array(dftsla['Close']),"TSLA"]
def lasso_elastic_net(data):
    y_train = data[0][0:1001] 
    X_train = np.linspace(0, 1000, 1001).reshape(-1, 1) 
    X_test = np.linspace(1001, 1500, 500).reshape(-1, 1)
    y_test = data[0][1001:1501]
    
    alpha = 0.1
    lasso = Lasso(alpha=alpha)

    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    print(lasso)
    print("r^2 on test data : %f" % r2_score_lasso)


    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    r2_score_enet = r2_score(y_test, y_pred_enet)
    print(enet)
    print("r^2 on test data : %f" % r2_score_enet)

    m, s, _ = plt.stem(np.where(enet.coef_)[0], enet.coef_[enet.coef_ != 0], markerfmt='x', label='Elastic net coefficients', use_line_collection=True)
    plt.setp([m, s], color="#2ca02c")
    m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],markerfmt='x', label='Lasso coefficients',use_line_collection=True)
    plt.setp([m, s], color='#ff7f0e')
    plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',markerfmt='bx', use_line_collection=True)

    plt.legend(loc='best')
    plt.title(data[1]+" Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"% (abs(r2_score_lasso), abs(r2_score_enet)))
    plt.show()


# In[22]:


lasso_elastic_net(ddata1)


# In[23]:


lasso_elastic_net(ddata2)


# In[27]:


#decision tree function
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
tdata1 = [np.array(dffb['Open']),"FB"]
tdata2 = [np.array(dftsla['Open']),"TSLA"]
def decision_tree_regression(data):
    X = np.linspace(0, 99, 100).reshape(-1, 1)
    y = data[0][0:100]
    #print(X)
    #print(y)

    regr_1 = DecisionTreeRegressor(max_depth=5)
    regr_2 = DecisionTreeRegressor(max_depth=7)
    regr_3 = DecisionTreeRegressor(max_depth=10)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    X_test = np.arange(0.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)

    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=5", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=7", linewidth=2)
    #plt.plot(X_test, y_3, color="red", label="max_depth=10", linewidth=2)
    plt.xlabel("trading days")
    plt.ylabel("stock price")
    plt.title(data[1]+" Decision Tree Regression")
    plt.legend()
    plt.show()


# In[28]:


decision_tree_regression(tdata1)


# In[29]:


decision_tree_regression(tdata2)


# In[54]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def f(x):
    return x * np.sin(x)

def polyRegression(data1,data2,name1,name2,num):
    # generate points used to plot
    print(np.size(data1))
    x_plot=np.linspace(0,num,10*num+1)

    # generate points and keep a subset of them
    x=data1
    y=data2
    print(x)
    print(y)
    print(np.newaxis)

    # create matrix versions of these arrays
    X = x[:, np.newaxis]
    X_plot = x_plot[:, np.newaxis]

    colors = ['teal', 'yellowgreen', 'gold']
    lw = 2
    plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
             label="ground truth")
    plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")
    plt.xlabel(name1)
    plt.ylabel(name2)

    for count, degree in enumerate([2,3,4]):
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        y_plot = model.predict(X_plot)
        plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,label="degree %d" % degree)

    plt.legend(loc='lower left')

    plt.show()


# In[55]:


data1=np.array(dffb['Delta'])
name1="FB abs Delta"
data2=np.array(dffb['maxDelta'])
name2="FB max abs Delta"
polyRegression(data1,data2,name1,name2,30)


# In[56]:


data1=np.array(dffb['Delta'])
name1="FB abs Delta"
data2=np.array(dffb['Volume'])
name2="FB Volume"
polyRegression(data1,data2,name1,name2,30)


# In[57]:


data1=np.linspace(0,1510,1511)
name1="trading days"
data2=np.array(dffb["Volume"])
name2="FB Volume"
polyRegression(data1,data2,name1,name2,2000)


# In[47]:


data1=np.linspace(0,1000,1001)
name1="trading days"
data2=np.array(dffb["Volume"][:1001])
name2="FB Volume"
polyRegression(data1,data2,name1,name2,1500)


# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

def PCR_PLS(data1,data2,name1,name2,company_name):
    X_train = data1[0:1000].reshape(-1, 1)
    y_train = data2[0:1000] 
    X_test = data1[1000:1200].reshape(-1, 1)
    y_test = data2[1000:1200]
    pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
    pcr.fit(X_train, y_train)
    pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline

    pls = PLSRegression(n_components=1)
    pls.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].scatter(pca.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[0].scatter(pca.transform(X_test), pcr.predict(X_test), alpha=.3,label='predictions')
    axes[0].set(xlabel=name1,ylabel=name2, title=company_name+' PCR')
    axes[0].legend()
    axes[1].scatter(pls.transform(X_test), y_test, alpha=.3, label='ground truth')
    axes[1].scatter(pls.transform(X_test), pls.predict(X_test), alpha=.3,label='predictions')
    axes[1].set(xlabel=name1,ylabel=name2, title=company_name+' PLS')
    axes[1].legend()
    plt.tight_layout()
    plt.show()


# In[61]:


data1=np.linspace(0,1510,1511)
name1="trading days"
data2=np.array(dffb["Volume"])
name2="FB Volume"
PCR_PLS(data1,data2,name1,name2,"FB")


# In[62]:


data1=np.array(dffb['Delta'])
name1="FB Delta"
data2=np.array(dffb['maxDelta'])
name2="FB max Delta"
PCR_PLS(data1,data2,name1,name2,"FB")


# In[58]:


data1=np.array(dftsla['Delta'])
name1="TSLA Delta"
data2=np.array(dftsla['maxDelta'])
name2="TSLA max Delta"
PCR_PLS(data1,data2,name1,name2,"TSLA")


# In[63]:


data1=np.array(dffb['Delta'])
name1="TSLA Delta"
data2=np.array(dffb['Volume'])
name2="TSLA Volume"
PCR_PLS(data1,data2,name1,name2,"TSLA")

