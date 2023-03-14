from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import pandas as pd
import seaborn as sns
from matplotlib import rc
from sklearn import linear_model
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import random
import streamlit as st
##################################################################################################################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
##################################################################################################################################################################

st.markdown(""" <style> .font_title {
font-size:50px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_header {
font-size:50px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subheader {
font-size:35px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subsubheader {
font-size:28px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_text {
font-size:26px ; font-family: 'times';text-align: left;} 
</style> """, unsafe_allow_html=True)

st.markdown(""" <style> .font_subtext {
font-size:18px ; font-family: 'times';text-align: center;} 
</style> """, unsafe_allow_html=True)

font_css = """
<style>
button[data-baseweb="columns"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px; font-family: 'times';
}
</style>
"""
# def sfmono():
    # font = "Times"
    
    # return {
        # "config" : {
             # "title": {'font': font},
             # "axis": {
                  # "labelFont": font,
                  # "titleFont": font
             # }
        # }
    # }

# alt.themes.register('sfmono', sfmono)
# alt.themes.enable('sfmono')
####################################################################################################################################################################
st.markdown('<p class="font_title">Chapter 4 - Part 3: Optimization</p>', unsafe_allow_html=True)
cols = st.columns(2,gap='small')
cols[0].image("https://editor.analyticsvidhya.com/uploads/58182variations_comparison.png")
cols[1].image("https://www.baeldung.com/wp-content/uploads/sites/4/2022/01/batch-1-1024x670.png")
####################################################################################################################################################################
st.markdown('<p class="font_text"> Lets see how optimization method affect weight calculation for a fake linear dataset with noise.</p>', unsafe_allow_html=True)
####################################################################################################################################################################
def GD(X, y, lr, epoch, m, b):  
    log, mse = [], []
    N = len(X)
    for _ in range(epoch): 
        if _ == 0:
            log.append((m, b))
        f = y - (m*X + b)
        m -= lr * (-2 * X.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    return m, b, log, mse
####################################################################################################################################################################
def S_MB_GD(X, y, lr, epoch, m, b, batch_size):
    log, mse = [], []
    for _ in range(epoch):
        if _ == 0:
            log.append((m, b))
        indexes = np.random.randint(0, len(X), batch_size) # random sample
        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)
        f = ys - (m*Xs + b)
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)
        log.append((m, b))
        mse.append(mean_squared_error(y, m*X+b))        
    return m, b, log, mse
####################################################################################################################################################################
Sample_Number = st.slider('Data size:', 20, 200, value=40)
cols1 = st.columns(2,gap='small')
####################################################################################################################################################################
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=False)
####################################################################################################################################################################
# X_Min = st.sidebar.slider('Minimum value of X:', -20, 20, value=0)
# X_Max = st.sidebar.slider('Maximum value of X:', X_Min, 100, value=20)
# Noise_Max = st.sidebar.slider('Maximum amount of noise:', 0, 40, value=1)
# Seed_Number = st.sidebar.slider('Seed number:', 1, 200, value=10)
np.random.seed(10)
X = np.random.uniform(0, 20,Sample_Number)
Noise = np.random.uniform(-1, 1,Sample_Number)
Initial_Slope = 4#st.sidebar.number_input('Slope of fake data:', min_value=-10.0, max_value=10.0,step=0.01, value=2.0,format='%f')
Initial_Intercept = 8#st.sidebar.number_input('Intercept of fake data:', min_value=-10.0, max_value=10.0,step=0.01, value=5.0,format='%f')
y=X*Initial_Slope+Noise+Initial_Intercept
####################################################################################################################################################################
Data_Visualization = cols1[0].checkbox('Visualize Fake Data?')
if Data_Visualization:
    Fig1,ax=plt.subplots(figsize=(12,10))
    plt.scatter(X,y)
    plt.xlabel("X",fontsize=25)
    plt.ylabel("y",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Data Visualization',fontsize=25)
    plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.5)
    cols1[0].pyplot(Fig1)
####################################################################################################################################################################
Gradient_Descent = cols1[1].checkbox('Performing Gradient Descent?')
if Gradient_Descent:
    Fig,ax=plt.subplots(figsize=(12,10))
    Slope=np.linspace(-10,10,200)#np.round(np.linspace(-0.2,0.2,200),2)
    Intecept=np.linspace(-10,10,200)#np.round(np.linspace(-0.1,0.1,200),2)
    Slope_Cont,Intercept_Cont= np.meshgrid(Slope, Intecept)
    Loss_MSE=np.zeros_like(Slope_Cont)

    for i in range (0,Intecept.shape[0]):
        for j in range (0,Slope.shape[0]):
            Loss_MSE[i,j] =mean_squared_error(y,Intercept_Cont[i,j]+Slope_Cont[i,j]*X)

    im=plt.contourf(Intercept_Cont, Slope_Cont, np.log(Loss_MSE), 1000, cmap='ocean')
    plt.xlabel("Intercept",fontsize=25)
    plt.ylabel("Slope",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='Log(MSE)',fontsize=20)
    epoch = cols1[1].slider('Number of epoch GD:', 200, 200000, value=5000)
    learning_rate = cols1[1].number_input('Learning rate GD:', min_value=0.0, max_value=0.01,step=0.005, value=0.0001,format='%f')
    Slope_Old = cols1[1].number_input('Initial slope GD:', min_value=-10, max_value=10,step=1, value=4,format='%i')
    Intercept_Old = cols1[1].number_input('Initial intercept GD:', min_value=-10, max_value=10,step=1, value=-2,format='%i')
    m, b, log, mse=GD(X,y,0.005,epoch,Slope_Old,Intercept_Old)
    log = np.array(log)
    plt.plot(log[:,1],log[:,0],alpha=0.7,color='red')
    plt.scatter(Initial_Intercept,Initial_Slope,s=40,color='yellow')
    plt.title('Gradient Descent',fontsize=25)
    plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.7)
    cols1[1].pyplot(Fig)
####################################################################################################################################################################
cols2 = st.columns(2,gap='small')
Batch_Gradient_Descent = cols2[0].checkbox('Performing Batch Gradient Descent?')
if Batch_Gradient_Descent:
    Fig,ax=plt.subplots(figsize=(12,10))
    Slope=np.linspace(-10,10,200)#np.round(np.linspace(-0.2,0.2,200),2)
    Intecept=np.linspace(-10,10,200)#np.round(np.linspace(-0.1,0.1,200),2)
    Slope_Cont,Intercept_Cont= np.meshgrid(Slope, Intecept)
    Loss_MSE=np.zeros_like(Slope_Cont)

    for i in range (0,Intecept.shape[0]):
        for j in range (0,Slope.shape[0]):
            Loss_MSE[i,j] =mean_squared_error(y,Intercept_Cont[i,j]+Slope_Cont[i,j]*X)

    im=plt.contourf(Intercept_Cont, Slope_Cont, np.log(Loss_MSE), 1000, cmap='ocean')
    plt.xlabel("Intercept",fontsize=25)
    plt.ylabel("Slope",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='Log(MSE)',fontsize=20)
    epoch = cols2[0].slider('Number of epoch BGD:', 200, 200000, value=5000)
    learning_rate = cols2[0].number_input('Learning rate BGD:', min_value=0.0, max_value=0.01,step=0.005, value=0.0001,format='%f')
    batch_size = cols2[0].slider('Batch Sizen BGD:', 1, len(X), value=10)
    Slope_Old = cols2[0].number_input('Initial slope BGD:', min_value=-10, max_value=10,step=1, value=4,format='%i')
    Intercept_Old = cols2[0].number_input('Initial intercept BGD:', min_value=-10, max_value=10,step=1, value=-2,format='%i')
    m, b, log, mse=S_MB_GD(X,y,0.005,epoch,Slope_Old,Intercept_Old,batch_size)
    log = np.array(log)
    plt.plot(log[:,1],log[:,0],alpha=0.7,color='red')
    plt.scatter(Initial_Intercept,Initial_Slope,s=40,color='yellow')
    plt.title('Batch Gradient Descent',fontsize=25)
    plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.7)
    cols2[0].pyplot(Fig)
####################################################################################################################################################################
Stochastic_Gradient_Descent = cols2[1].checkbox('Performing Stochastic Gradient Descent?')
if Stochastic_Gradient_Descent:
    Fig,ax=plt.subplots(figsize=(12,10))
    epoch = cols2[1].slider('Number of epoch SGD:', 200, 200000, value=5000)
    learning_rate = cols2[1].number_input('Learning rate SGD:', min_value=0.0, max_value=0.01,step=0.005, value=0.0001,format='%f')
    Slope_Old = cols2[1].number_input('Initial slope SGD:', min_value=-10, max_value=10,step=1, value=4,format='%i')
    Intercept_Old = cols2[1].number_input('Initial intercept SGD:', min_value=-10, max_value=10,step=1, value=-2,format='%i')
    m, b, log, mse=S_MB_GD(X,y,0.005,epoch,Slope_Old,Intercept_Old,1)
    log = np.array(log)
    
    Slope=   np.linspace(int(np.min(log[:,0]))-2,int(np.max(log[:,0]))+2,200)#np.round(np.linspace(-0.2,0.2,200),2)
    Intecept=np.linspace(int(np.min(log[:,1]))-2,int(np.max(log[:,1]))+2,200)#np.round(np.linspace(-0.1,0.1,200),2)
    Slope_Cont,Intercept_Cont= np.meshgrid(Slope, Intecept)
    Loss_MSE=np.zeros_like(Slope_Cont)

    for i in range (0,Intecept.shape[0]):
        for j in range (0,Slope.shape[0]):
            Loss_MSE[i,j] =mean_squared_error(y,Intercept_Cont[i,j]+Slope_Cont[i,j]*X)

    im=plt.contourf(Intercept_Cont, Slope_Cont, np.log(Loss_MSE), 1000, cmap='ocean')
    plt.xlabel("Intercept",fontsize=25)
    plt.ylabel("Slope",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='Log(MSE)',fontsize=20)
    
    plt.plot(log[:,1],log[:,0],alpha=0.7,color='red')
    plt.scatter(Initial_Intercept,Initial_Slope,s=40,color='yellow')
    plt.title('Stochastic Gradient Descent',fontsize=25)
    plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.7)
    cols2[1].pyplot(Fig)
####################################################################################################################################################################
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013). </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">2) https://www.analyticsvidhya.com/blog/2022/07/gradient-descent-and-its-types/</p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">3) https://www.baeldung.com/cs/learning-rate-batch-size</p>', unsafe_allow_html=True)
##################################################################################################################################################################
