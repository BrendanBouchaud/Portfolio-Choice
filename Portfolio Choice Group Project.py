# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        Group Project - Financial Instrument & Portfolio Choices
            Momentum (Cumulative return over the past 12 months)


            Created on Sat Dec 21 18:40:00 2020


                        @authors:
                     
                   - Mouhaned Bin Yousef
                   - Brendan BOUCHAUD
                   - Thomas GRIMAULT
                   - Elias ESSAADOUNI
    
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%% Modules

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as smf


import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()


#%%

"""
    ======================================================================================================================
    ================================================            1             ============================================
    ======================================================================================================================
    Import data un process it
"""

#%% Import datasets : Stock prices and identifiers

path = os.path.dirname(os.path.abspath("__file__"))
df_extract = pd.read_csv(path + r"/Export_CRSP.csv")
ff_extract = pd.read_csv(path + r"/F-F_Research_Data_5_Factors_2x3.CSV", skiprows = 0, index_col = 0)


#%% Transform ff = farma french data set
ff = ff_extract[ff_extract.iloc[:,0].isna().cumsum() == 0] #Clean all annual data (after empty row)
ff[['Mkt-RF','SMB','HML','RMW','CMA','RF']] = ff[['Mkt-RF','SMB','HML','RMW','CMA','RF']].apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
ff['Mkt returns'] = ff['Mkt-RF'] + ff['RF']
ff.index.names = ['date']

#%% Transform df = export CRSP
df = df_extract.copy()

#Data correction 1 : date to str
df['date']=(df['date']).apply(str)
df['date']=df['date'].str[0:6]

#Data Correction 2 : Date, price, vol, return, share to numeric
df[['date','PRC','VOL','RET','SHROUT']]= df[['date','PRC','VOL','RET','SHROUT']].apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer') #Transform from data type object to float

#Data Correction 3 : Rename columns
df = df.rename(columns={'SHRCD': 'Shares Code','EXCHCD': 'Exchange Code','PRC':'Price', 'VOL':'Volatility', 'RET':'Return', 'SHROUT': 'Shares Outstanding'})#rename columns

#Data Correction 4 : Reset index
df = df.reset_index(drop=True)#reset index to 0 

#Data Correction 5 : Negative Value
df['Price'] = df['Price'].abs()

#%% add Market Cap for each permno each date


df['Mk Cap'] = df['Shares Outstanding']*df['Price']


#%%Compute 12month cumulative return : with all data
"""
    ======================================================================================================================
    ================================================            2             ============================================
    ======================================================================================================================
    Calculate 12 months cumulative returns for each stocks in dataframe
"""


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!! TAKE 10 MIN !!! : 12 months cumulative returns calculation !!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

print('Calculating 12 Months Cummulative Returns...')
df['12 months Ret'] = df.groupby(['PERMNO'])['Return'].progress_apply(lambda x: (x+1).rolling(12).apply(np.prod)-1)

#%% 
"""
    ======================================================================================================================
    ================================================            3             ============================================
    =====================================================================================================================
    Create the 10 portfolios bases on mommentum quantiles 
"""
df_q = df.copy()

#Add Quantile number for each date of 12MR to data set df_q
df_q['q'] = df.groupby(['date'])['12 months Ret'].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')+1)


#%% Get Each Decile data in dataframe
portfolio_10_data = df_q.loc[(df_q['q']==10)]
portfolio_9_data = df_q.loc[(df_q['q']==9)]
portfolio_8_data = df_q.loc[(df_q['q']==8)]
portfolio_7_data = df_q.loc[(df_q['q']==7)]
portfolio_6_data = df_q.loc[(df_q['q']==6)]
portfolio_5_data = df_q.loc[(df_q['q']==5)]
portfolio_4_data = df_q.loc[(df_q['q']==4)]
portfolio_3_data = df_q.loc[(df_q['q']==3)]
portfolio_2_data = df_q.loc[(df_q['q']==2)]
portfolio_1_data = df_q.loc[(df_q['q']==1)]

#%% Function : Construct portfolios by calculating metrics for each months for each decine
def Portfolio_Processing(portfolio_input):
    portfolio = portfolio_input.copy() 
    portfolio['nb stocks'] = portfolio.groupby(['date'])['PERMNO'].transform("count")
    portfolio['Total MkCap'] = portfolio.groupby(['date'])['Mk Cap'].transform("sum")
    
    #1 Equal Weight
    portfolio['Eq Weight'] = 1 / portfolio['nb stocks']
    
    #2 Value Weight :
    portfolio['Val Weight'] = portfolio['Mk Cap']/portfolio['Total MkCap']
    
    #Calculate Weigted returns
    portfolio['eq w ret'] = portfolio['Eq Weight'] * portfolio['Return']
    portfolio['val w ret'] = portfolio['Val Weight'] * portfolio['Return']
    
    #Group returns by date
    portfolio_month = portfolio.groupby(['date'])[['eq w ret']].agg('sum')
    portfolio_month['val w ret'] = portfolio.groupby(['date'])[['val w ret']].agg('sum')

    return portfolio_month


#%% Calculate Each portfolio returns for the whole sample period usung Portfolio_Processing Function
portfolio_10_month = Portfolio_Processing(portfolio_10_data)
portfolio_9_month = Portfolio_Processing(portfolio_9_data)
portfolio_8_month = Portfolio_Processing(portfolio_8_data)
portfolio_7_month = Portfolio_Processing(portfolio_7_data)
portfolio_6_month = Portfolio_Processing(portfolio_6_data)
portfolio_5_month = Portfolio_Processing(portfolio_5_data)
portfolio_4_month = Portfolio_Processing(portfolio_4_data)
portfolio_3_month = Portfolio_Processing(portfolio_3_data)
portfolio_2_month = Portfolio_Processing(portfolio_2_data)
portfolio_1_month = Portfolio_Processing(portfolio_1_data)

#%% Create Portfolio Long Short
portfolio_long_short = pd.DataFrame(
    {'eq w ret': portfolio_10_month['eq w ret']-portfolio_1_month['eq w ret'], 
     'val w ret':portfolio_10_month['val w ret']-portfolio_1_month['val w ret']})


#%% Concatenate all portfolio returns
"""
    ======================================================================================================================
    ================================================           4              ============================================
    ======================================================================================================================
     Concatenate all portfolio returns  & export to excel
"""
all_portfolios_month_returns = pd.concat([portfolio_10_month, 
                            portfolio_9_month,
                            portfolio_8_month,
                            portfolio_7_month,
                            portfolio_6_month,
                            portfolio_5_month,
                            portfolio_4_month,
                            portfolio_3_month,
                            portfolio_2_month,
                            portfolio_1_month,
                            portfolio_long_short,
                            ], axis=1)

#%% Change Columns names to match portfolios Q & Weighted Return Types

all_portfolios_columns_name = ['Ptf 10: Equal Weighted Returns', 'Ptf 10: Value Weighted Returns', 
                         'Ptf 09: Equal Weighted Returns', 'Ptf 09: Value Weighted Returns', 
                         'Ptf 08: Equal Weighted Returns', 'Ptf 08: Value Weighted Returns', 
                         'Ptf 07: Equal Weighted Returns', 'Ptf 07: Value Weighted Returns', 
                         'Ptf 06: Equal Weighted Returns', 'Ptf 06: Value Weighted Returns', 
                         'Ptf 05: Equal Weighted Returns', 'Ptf 05: Value Weighted Returns', 
                         'Ptf 04: Equal Weighted Returns', 'Ptf 04: Value Weighted Returns', 
                         'Ptf 03: Equal Weighted Returns', 'Ptf 03: Value Weighted Returns', 
                         'Ptf 02: Equal Weighted Returns', 'Ptf 02: Value Weighted Returns', 
                         'Ptf 01: Equal Weighted Returns', 'Ptf 01: Value Weighted Returns', 
                         'Ptf L/S: Equal Weighted Returns', 'Ptf L/S: Value Weighted Returns',
                         ]

all_portfolios_month_returns = all_portfolios_month_returns.set_axis(all_portfolios_columns_name, axis='columns', inplace=False)

#%%
# Calculate Mean 12 months Retruns for each portfolio
all_portfolios_month = all_portfolios_month_returns.copy()
all_portfolios_month += 1
all_portfolios_month = all_portfolios_month.reset_index()
all_portfolios_month['year'] = all_portfolios_month['date'].astype(str).str[0:4].astype(int)
all_portfolios_month = all_portfolios_month.set_index('year')


# Calculate Mean Annual Retruns whole period
all_portfolios_annual_returns = all_portfolios_month.groupby(['year'])[all_portfolios_columns_name].cumprod() - 1 #calculate cum returns on the year for each stocks
all_portfolios_annual_returns['date'] = all_portfolios_month['date']
all_portfolios_annual_returns_y = all_portfolios_annual_returns.groupby(['year'])[all_portfolios_columns_name].max() #get the last raw of cum ret / years to get the annual return of the ptfs
all_portfolios_annual_returns_mean = all_portfolios_annual_returns_y.mean(axis=0) * 100
all_portfolios_annual_returns_mean = pd.DataFrame(all_portfolios_annual_returns_mean).reset_index()
all_portfolios_annual_returns_mean.columns = ['Portfolio', 'Raw Return']
all_portfolios_annual_returns_mean = all_portfolios_annual_returns_mean.set_index('Portfolio')

#%%
"""
    ======================================================================================================================
    ================================================            5             ============================================
    ======================================================================================================================
     Calculate For each portfolio Beta, Alpha, FF3 factors & FF5 factors 
"""

# Merging the ff data with portfolio based on time index (%Y%m)

def alpha_data_processing(portfolio):
    ptf_data = portfolio.copy() * 100 # Display as % like ff file
    ptf_data.index = ptf_data.index.map(str) #convert to str for merging
    data_concat = pd.concat([ff,ptf_data], axis=1)
    data_concat = data_concat.dropna(how='any',axis=0)
    # Rename the columns
    data_concat.rename(columns={'Mkt-RF':'mkt_excess', 'val w ret': "port_ValW_returns", 'eq w ret': "port_EqW_returns"}, inplace=True)
    # Calculate the excess returns
    data_concat['port_ValW_excess'] = data_concat['port_ValW_returns'] - data_concat['RF']
    data_concat['port_EqW_excess'] = data_concat['port_EqW_returns'] - data_concat['RF']
    return data_concat
                      
# Calculate CAPM 
def CAPM(portfolio):
    data = alpha_data_processing(portfolio)
    
    #Value Weighted Beta & Alpha CAPM
    (beta_v, alpha_v) = stats.linregress(data['mkt_excess'].values,
                    data['port_EqW_returns'].values)[0:2]
    (beta_e, alpha_e) = stats.linregress(data['mkt_excess'].values,
                    data['port_ValW_returns'].values)[0:2]
    
    beta = pd.DataFrame({'Beta Val w': [beta_v], 'Beta Eq w': [beta_e]})
    alpha = pd.DataFrame({'Alpha Val w': [alpha_v], 'Aplha Eq w': [alpha_e]})
    
    return beta, alpha

# FAMA FRENCH 3F Alpha
def FF3(portfolio):
    data = alpha_data_processing(portfolio)
    model_e = smf.formula.ols(formula = "port_EqW_returns ~ mkt_excess + SMB + HML", data = data).fit()
    param_e = model_e.params 
    model_v = smf.formula.ols(formula = "port_ValW_returns ~ mkt_excess + SMB + HML", data = data).fit()
    param_v = model_v.params 
    FF3_Factors = pd.concat([param_e, param_v], axis = 1)  
    FF3_Factors = FF3_Factors.set_axis(['Eq w FF3','Val w FF3'], axis=1, inplace=False)      
    return FF3_Factors

# FAMA FRENCH 5F Aplha
def FF5(portfolio):
    data = alpha_data_processing(portfolio)
    model_e = smf.formula.ols(formula = "port_EqW_returns ~ mkt_excess + SMB + HML + RMW + CMA", data = data).fit()
    param_e = model_e.params 
    model_v = smf.formula.ols(formula = "port_ValW_returns ~ mkt_excess + SMB + HML + RMW + CMA", data = data).fit()
    param_v = model_v.params
    
    FF5_Factors = pd.concat([param_e, param_v], axis = 1)   
    FF5_Factors = FF5_Factors.set_axis(['Eq w FF5','Val w FF5'], axis=1, inplace=False)
    return FF5_Factors

#%% Get Alphas, Betas and Fama french factors

beta_10, alpha_10 = CAPM(portfolio_10_month)
ff3_10 = FF3(portfolio_10_month)
ff5_10 = FF5(portfolio_10_month)

beta_9, alpha_9 = CAPM(portfolio_9_month)
ff3_9 = FF3(portfolio_9_month)
ff5_9 = FF5(portfolio_9_month)

beta_8, alpha_8 = CAPM(portfolio_8_month)
ff3_8 = FF3(portfolio_8_month)
ff5_8 = FF5(portfolio_8_month)

beta_7, alpha_7 = CAPM(portfolio_7_month)
ff3_7 = FF3(portfolio_7_month)
ff5_7 = FF5(portfolio_7_month)

beta_6, alpha_6 = CAPM(portfolio_6_month)
ff3_6 = FF3(portfolio_6_month)
ff5_6 = FF5(portfolio_6_month)

beta_5, alpha_5 = CAPM(portfolio_5_month)
ff3_5 = FF3(portfolio_5_month)
ff5_5 = FF5(portfolio_5_month)

beta_4, alpha_4 = CAPM(portfolio_4_month)
ff3_4 = FF3(portfolio_4_month)
ff5_4 = FF5(portfolio_4_month)

beta_3, alpha_3 = CAPM(portfolio_3_month)
ff3_3 = FF3(portfolio_3_month)
ff5_3 = FF5(portfolio_3_month)

beta_2, alpha_2 = CAPM(portfolio_2_month)
ff3_2 = FF3(portfolio_2_month)
ff5_2 = FF5(portfolio_2_month)

beta_1, alpha_1 = CAPM(portfolio_1_month)
ff3_1 = FF3(portfolio_1_month)
ff5_1 = FF5(portfolio_1_month)

beta_LS, alpha_LS = CAPM(portfolio_long_short)
ff3_LS = FF3(portfolio_long_short)
ff5_LS = FF5(portfolio_long_short)

# Regroup Raw Returns, Betas & Factor in the same table Columns : Portfolios (W & Val) 1, 2, 3... L/S
alphas = pd.concat([alpha_10.T, alpha_9.T, alpha_8.T, alpha_7.T, alpha_6.T, alpha_5.T, alpha_4.T, alpha_3.T, 
                    alpha_2.T, alpha_1.T, alpha_LS.T], axis= 0).set_axis(all_portfolios_columns_name, axis = 0)
alphas.columns = ['CAPM Alpha']

betas = pd.concat([beta_10.T, beta_9.T, beta_8.T, beta_7.T, beta_6.T, beta_5.T, beta_4.T, beta_3.T, 
                   beta_2.T, beta_1.T, beta_LS.T], axis=0).set_axis(all_portfolios_columns_name, axis = 0)
betas.columns = ['CAPM Beta']

ff3 = pd.concat([ff3_10.T,ff3_9.T,ff3_8.T,ff3_7.T, ff3_6.T, ff3_5.T, ff3_4.T, ff3_3.T, ff3_2.T, ff3_1.T, 
                 ff3_LS.T], axis=0).set_axis(all_portfolios_columns_name, axis = 0)
ff3.columns = ['Intercept FF3', 'mkt_excess FF3', 'SMB FF3', 'HML FF3']
ff5 = pd.concat([ff5_10.T,ff5_9.T,ff5_8.T,ff5_7.T, ff5_6.T, ff5_5.T, ff5_4.T, ff5_3.T, ff5_2.T, ff5_1.T, 
                 ff5_LS.T], axis=0).set_axis(all_portfolios_columns_name, axis = 0)
ff5.columns = ['Intercept FF5', 'mkt_excess FF5', 'SMB FF5', 'HML FF5', 'RMW FF5', 'CMA FF5']

all_data =  pd.concat([all_portfolios_annual_returns_mean, alphas, betas, ff3, ff5], axis=1)

all_data = all_data.reset_index()


"""
    ======================================================================================================================
    ======================================================================================================================
    ======================================================================================================================
"""

#%% Export to Excel
all_data.to_excel(path + r"/all_data.xlsx", encoding='utf-8', index=False)

