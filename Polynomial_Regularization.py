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
st.markdown('<p class="font_title">Chapter 4 - Part 2: Regularization for Polynomial Fitting</p>', unsafe_allow_html=True)
####################################################################################################################################################################
cols = st.columns([2, 2 , 2])
cols[0].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_ols_001.png")
cols[1].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_ridge_path_001.png")
cols[2].image("https://scikit-learn.org/stable/_images/sphx_glr_plot_lasso_lars_ic_001.png")
####################################################################################################################################################################
st.markdown('<p class="font_text"> Continuing the last ICA, now we investigate the importance of regularization for fitting a polynomial.</p>', unsafe_allow_html=True)
####################################################################################################################################################################
Dataset_Name = st.sidebar.selectbox('Select your dataset',('Sin(X)', 'X*Sin(X)', 'Sinh(X)', 'Exp(X)', 'X^N'),index = 4)
np.random.seed(10)
Data_Numbers = st.sidebar.slider('Size of fake data:', 5, 200, value=30)
X = np.random.uniform(0, 20,Data_Numbers).reshape(-1,1)
Noise = np.random.uniform(-1, 1,Data_Numbers).reshape(-1,1)
if Dataset_Name == 'Sin(X)':
    y=5*np.sin(X/2)+0.5*Noise
elif Dataset_Name == 'X*Sin(X)':
    y=X*np.sin(X/2)+0.5*Noise
elif Dataset_Name == 'Sinh(X)':
    y=np.sinh(X)+Noise
elif Dataset_Name == 'Exp(X)':
    y=2*np.exp(X)+Noise
else:
    y=3.6*X**0.5+Noise
####################################################################################################################################################################
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=False)
####################################################################################################################################################################
cols=st.columns([4,1])
Degree = st.sidebar.slider('Order of fitted polynomial:', 2, len(X), value=5)
X_Train = PolynomialFeatures(Degree).fit_transform(X.reshape(-1,1))
Fig1,ax=plt.subplots(figsize=(12,10))
plt.scatter(X,y,label='Fake Data')    
####################################################################################################################################################################
X_Lin = np.linspace(0,20,400)
Linear_Object = linear_model.LinearRegression(fit_intercept=True)
Linear_Object.fit(X_Train,y)
Coeff_Lin = Linear_Object.coef_
Y_Lin = 0
for count, value in enumerate(Coeff_Lin[0]):
    Y_Lin += value* X_Lin**count
plt.plot(X_Lin,Y_Lin+Linear_Object.intercept_,label='No Penalty',color='magenta',linestyle = 'solid',linewidth=3,alpha=0.5)    
####################################################################################################################################################################
Lasso = st.sidebar.checkbox('Add Lasso to Plot?')
if Lasso:
    Alpha_Lasso = st.sidebar.number_input('L1 penalty constant for Lasso', min_value=0.0, max_value=100000000.0,step=0.01, value=0.1,format='%f')
    Lasso_Object = linear_model.Lasso(fit_intercept=True, alpha=Alpha_Lasso,max_iter=10000000)
    Lasso_Object.fit(X_Train,y)
    Coeff_Lasso = Lasso_Object.coef_
    Y_Lin = 0
    for count, value in enumerate(Coeff_Lasso):
        Y_Lin += value* X_Lin**count
    plt.plot(X_Lin,Y_Lin+Lasso_Object.intercept_,label='L1 Penalty (Lasso)',color='green',linestyle = '-.',linewidth=3,alpha=0.5)  
####################################################################################################################################################################    
Ridge = st.sidebar.checkbox('Add Ridge to Plot?')
if Ridge:
    Alpha_Ridge = st.sidebar.number_input('L2 penalty constant for Ridge:', min_value=0.0, max_value=100000000.0,step=0.01, value=0.1,format='%f')
    Ridge_Object = linear_model.Ridge(fit_intercept=True, alpha=Alpha_Ridge,max_iter=10000000)
    Ridge_Object.fit(X_Train,y)
    Coeff_Ridge = Ridge_Object.coef_
    Y_Lin = 0
    for count, value in enumerate(Coeff_Ridge[0]):
        Y_Lin += value* X_Lin**count
    plt.plot(X_Lin,Y_Lin+Ridge_Object.intercept_,label='L2 Penalty (Ridge)',color='red',linestyle = 'dashed',linewidth=3,alpha=0.5)     
####################################################################################################################################################################
Elastic_Net = st.sidebar.checkbox('Add ElasticNet to Plot?')
if Elastic_Net:
    Alpha_Elastic_Net = st.sidebar.number_input('Constant for Elastic Net Penalty regression:', min_value=0.0, max_value=100000000.0,step=0.01, value=0.1,format='%f')
    L1_Ratio_Elastic_Net = st.sidebar.number_input('Value for Elastic Net mixing parameter for penalties:', min_value=0.0, max_value=1.0,step=0.01, value=0.5,format='%f')
    ElasticNet_Object = linear_model.ElasticNet(fit_intercept=True, alpha=Alpha_Elastic_Net, l1_ratio=L1_Ratio_Elastic_Net,max_iter=10000000)
    ElasticNet_Object.fit(X_Train,y)
    Coeff_ElasticNet = ElasticNet_Object.coef_
    Y_Lin = 0
    for count, value in enumerate(Coeff_ElasticNet):
        Y_Lin += value* X_Lin**count
    plt.plot(X_Lin,Y_Lin+ElasticNet_Object.intercept_,label='L1+L2 Penalty (ElasticNet)',color='orange',linestyle = '--',linewidth=3,alpha=0.5)
####################################################################################################################################################################   
plt.xlabel("X",fontsize=25)
plt.ylabel("y",fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.5)
cols[0].pyplot(Fig1)
####################################################################################################################################################################
st.markdown('<p class="font_header">References: </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">1) Buitinck, Lars, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae et al. "API design for machine learning software: experiences from the scikit-learn project." arXiv preprint arXiv:1309.0238 (2013). </p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">2) https://www.analyticsvidhya.com/blog/2022/07/gradient-descent-and-its-types/</p>', unsafe_allow_html=True)
st.markdown('<p class="font_text">3) https://www.baeldung.com/cs/learning-rate-batch-size</p>', unsafe_allow_html=True)
##################################################################################################################################################################
