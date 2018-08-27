# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 23:48:58 2018

@author: RBS
"""





import pandas as pd  # import pandas module for DataFrame, File reading and other functions
import datetime # import datatime for handling Dates
import numpy as np


import math
import fix_yahoo_finance as yf
import random
from scipy.cluster.vq import kmeans,vq
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import time


yf.pdr_override() 



def download_data(tickers, startDate, endDate):
    '''
    This functions takes the list of tickers and downloads data from Yahoo Finance from startDate to endDate
    '''
    quotes=yf.download(tickers,startDate, endDate)
    print(tickers)
    return quotes


def create_clusters(datas, volumes, counter):
    
    '''
    This function creates 5 clusters from the datas DF based on the annual return and volatility of the stocks
    
    '''
    datas.to_csv('dats.csv')
    volumes.to_csv('vols.csv')
    
    
    datas = datas.fillna(method='ffill')
    datas = datas.fillna(method='bfill')
    
    volumes = volumes.fillna(method='ffill')
    volumes = volumes.fillna(method='bfill')
    
    
    
    #Calculate average annual percentage return and volatilities over a theoretical one year period
    rets = datas.pct_change().mean() * 252
    
    rets = pd.DataFrame(rets)
    rets.columns = ['Returns']
    rets['Volatility'] = datas.pct_change().std() * sqrt(252)
    rets['SR']=rets['Returns']/rets['Volatility']
    
    
    liq=pd.DataFrame()
    liq_data=pd.DataFrame()
    
    volumes.to_csv('v.csv')
    datas.to_csv('d.csv')
    
    liq_data=volumes*datas.values
    
    liq['Volume']=liq_data.mean()*252
 
    #format the data as a numpy array to feed into the K-Means algorithm
    
    liq.to_csv('liq.csv')
    
    data = np.asarray([np.asarray(rets['Returns']),np.asarray(rets['Volatility']),np.asarray(liq['Volume'])]).T
 
    X = data
    
    df=pd.DataFrame(data=X)
    
    distorsions = []
    for k in range(2, 20):
        k_means = KMeans(n_clusters=k)
        k_means.fit(X)
        distorsions.append(k_means.inertia_)
 
    if counter==0:
        fig=plt.figure(1)
        plt.plot(range(2, 20), distorsions)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Squared Error')
        plt.grid(True)
        plt.title('Elbow curve')
        plt.show()        
        
    
    
    # computing K-Means with K = 5 (5 clusters)
    centroids,_ = kmeans(data,5)
    # assign each sample to a cluster
    idx,_ = vq(data,centroids)
    
    if counter==0:
        
        fig2=plt.figure(2)
        # some plotting using numpy's logical indexing
        plt.plot(data[idx==0,0],data[idx==0,1],'ob',
                 data[idx==1,0],data[idx==1,1],'oy',
                 data[idx==2,0],data[idx==2,1],'or',
                 data[idx==3,0],data[idx==3,1],'og',
                 data[idx==4,0],data[idx==4,1],'om', label='Clusters')
        
        plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, label='Cebtroids')
        plt.title("Stock Clusters with Centroids")
        plt.legend(loc='upper left')
        plt.show()
    
    
    cluster=pd.DataFrame(index=rets.index)
    cluster['Cluster']=idx
    
    
    return cluster
   
        
def riskreturn_profile(portfolio, pos_trade, prices):
    '''
    This functions calculates risk & retrns metrics for the trades data and portolfio series
    These are the mertics calculated
    
    Maximum Drawdown
    CAGR
    Lake Ratio
    Pain to Gain Ratio
    Standard Deviation
    Win Ratio
    Loss Ratio
    Average Win Return
    Average Loss Return
    
    '''
    profile=pd.Series(index=['CAGR','Standard_Deviation', 'MaxDrawdown','LakeRatio','GainPainRatio', 'Sharpe_Ratio', 'Win_Rate', 'Loss_Rate', 'Average_Win_Returns', 'Average_Loss_Returns'])
    
 
    
    #Calculate CAGR  
    numofyears=(((portfolio.index[len(portfolio.index)-1]-portfolio.index[0]).days)/365)
    
    if (portfolio.loc[portfolio.index[-1]]<0) or (portfolio.loc[portfolio.index[0]]<0):
        CAGR= (portfolio.loc[portfolio.index[-1]]/portfolio.loc[portfolio.index[0]]-1)/numofyears
    else:
        CAGR=math.pow((portfolio.loc[portfolio.index[-1]]/portfolio.loc[portfolio.index[0]]),(1/numofyears))-1
    
    profile.loc['CAGR']=CAGR
    
    p_ret=(portfolio/portfolio.shift(-1))-1
    profile.loc['Standard_Deviation']=p_ret.std()*sqrt(252)
    
    profile.loc['Sharpe_Ratio']=(CAGR-0.01)/(p_ret.std()*sqrt(252))
    
    #Calculate Max Drawdown
    MaxDrawdown=calcMaxDrawdown(portfolio.loc[:])
    
    #Calculate Lake Ratio
    lr=lakeRatio(portfolio.loc[:])
    
    #Calculate Gain-to-Pain Ratio
    gp=GPR(portfolio.loc[:])
    
    
    #Calculate the number of Trades done (when position does change vs previous day)
    trades=pos_trade-pos_trade.shift(-1)
    trades[trades<0]=1
    trades[trades>0]=1
    
    #calculate the daily returns as = Daily PnL/position size of previous day
    trade_ret= (pos_trade *(prices.shift(1)-prices))*trades
    trade_ret=trade_ret.dropna(axis=0, how='all')
    trade_ret=trade_ret.dropna(axis=1, how='all')
    trade_ret1=trade_ret
    pos_value=pos_trade*prices
    pos_value=pos_value.dropna(axis=0, how='all')
    pos_value=pos_value.dropna(axis=1, how='all')
    pos_value=pos_value[pos_value!=0]
    trade_ret1=trade_ret[trade_ret1!=0]
    
    
    trade_return=trade_ret1/pos_value
    
    trade_return=trade_return.fillna(0)
    
    #calculate the aveagre return for win trades (where pnl>0)
    win_rets=trade_return[trade_return>0]
    
    #calculate the average return for loss trades (where pnl<0)
    lose_rets=trade_return[trade_return<0]
    
    win_ret=win_rets.stack().dropna().mean()
    lose_ret=lose_rets.stack().dropna().mean()
    
    
    #calculate the win ratio and loss ratio
    win_trades=trade_ret
    win_trades[win_trades>0]=1
    win_trades[win_trades<0]=0
    win_trades=win_trades.fillna(0)
    n_win=win_trades.values.sum()
    trades=trades.fillna(0)
    n_trade=trades.values.sum()
    win_ratio=n_win/n_trade
    lose_ratio=1-win_ratio
    
    
    profile.loc['MaxDrawdown']=MaxDrawdown
    profile.loc['LakeRatio']=lr
    profile.loc['GainPainRatio']=gp    
    
    profile.loc['Win_Rate']=win_ratio    
    profile.loc['Loss_Rate']=lose_ratio    
    profile.loc['Average_Win_Returns']=win_ret    
    profile.loc['Average_Loss_Returns']=lose_ret    
    
    
    
    return profile

def calcMaxDrawdown(perf):
    '''
    This functions takes performance DataFrame and calculates Maximum Dradown
    
    '''
    
    globalmax=0.00001
    
    maxdrawdown=0.0001
    
    for i in range(0, len(perf)):
        if perf.iloc[i] > globalmax:
            globalmax=perf.iloc[i]
        drawdown=(globalmax-perf.iloc[i])/globalmax
        if drawdown > maxdrawdown:
            maxdrawdown=drawdown
    
    return maxdrawdown
            


def lakeRatio(data):
    '''
    This function calculates the Lake Ratio for our Strategy
    '''
    water=0.0
    earth=0.0
    maxs=0
    lakeRatios=0.0
    for row in data.index:
        if data[row] > maxs:
            maxs=data[row]
            
        dd=data[row]
        water=water+(maxs-dd)
        earth=earth+data[row]
    
    lakeRatios=water/earth
    return lakeRatios


def calc_sharpe(port):
    
    '''
    This function calculates Sharpe ratio for our Strategy
    '''
    
    ret=(port.loc[-1]/port.loc[0])-1
    stds=port.std()
    
    sharpe=ret/stds
    
    return sharpe


    
    
def GPR(port):
    '''
    This function calculates the Gain to Pain Ratio for our Strategy
    '''
    data=port.pct_change(1)
    pain=0
    gain=data.sum()
    for row in data.index:
        if data[row]<0:
            pain=pain+abs(data[row])
    
    GPratio=gain/pain
    
    return GPratio        


    
def calc_alphas2(returns, volumes, prices):
    '''
    This function calcultes alphas using this expression
    
    alphas=((Sum of 5D volume/Sum of 60D volume)*(1- 20D Standard_Deviation)*(-1* Sum of 5D Return))
    
    
    '''
        
 
    long_volume=volumes.rolling(window=60).sum()
    short_volume=volumes.rolling(window=5).sum()
    
    stds=returns.rolling(window=20).std()
    ma_ret=returns.rolling(window=5).sum()
    ma_ret=(-1*ma_ret)
    stds=stds.abs()
    stds=1-stds
    new_alphas= (short_volume/long_volume)*stds*ma_ret
    return new_alphas
    
    
def convert_weights(weights, limit):
    '''
    This functions takes a series of Weights and converts it so that total sum of the weights remains
    the same and none of the weight is greater than Limit
    Its used to set max weight of stocks in our alphas
    '''
    
    w=weights.abs()
    
    if len(w[weights>limit])==0:
        
        return weights
    
    else:
        
        if weights.sum()>0:
        
            diff=0
            weight=pd.Series(weights)
        
            for tick in weight.index:
        
                if weight.loc[tick]> limit:
            
                    diff=diff+(weight.loc[tick]-limit)
                    weight.loc[tick]=limit
            
    
            if diff>0:
                sums=weight[weight<limit].sum()
                new_weights=(weight[weight<limit]/sums)*diff
        
                for tick in new_weights.index:
                    weights.loc[tick]=weight.loc[tick]+new_weights.loc[tick]
    
        
            return convert_weights(weight,limit)
        
        else:
            
            weight=pd.Series(weights)
            
                    
            for tick in weight.index:
        
                if weight.loc[tick]< -1*limit:
            
                    diff=diff+(weight.loc[tick]-(-1*limit))
                    weight.loc[tick]=-1*limit
            
    
            if diff>0:
                sums=weight[weight>-1*limit].sum()
                new_weights=(weight[weight<limit]/sums)*diff
        
                for tick in new_weights.index:
                    weights.loc[tick]=weight.loc[tick]+new_weights.loc[tick]
    
        
            return convert_weights(weight,limit)
        
        
    

def decay_alpha(alphas, n):
    '''
    
    This function takes alphas dataframe and apply linear decay formula over n days
    '''
    
    count=0
    for date in alphas.index:
        
        location=alphas.index.get_loc(date)
        if count>=n:
            
            sum2=0
            for i in range(0,n):
                
                if i==0:
                    sums=alphas.iloc[location,:]*n
                    sum2=n-i
                else:
                    sums=sums+(alphas.iloc[location-i,:]*(n-i))
                    sum2=sum2+(n-i)
            alphas.loc[date,:]=sums/sum2
            
        count=count+1
                
    return alphas


def convert_to_usd(prices, fx, tickers, fxdata):
    '''
    
    This function takes prices and converts them to USD based on the FX rate in fxdata dataframe. The stocks are
    from different currencies and the relevant fx is looked up from fx dataframe
    '''
    usd_prices=prices
    
    
    
    for currency in tickers:
        stocks=fx[fx['FX']==currency]
        
        prevrate=fxdata.loc[fxdata.index[0],currency]
        
        for date in prices.index:
            
            list1=[date]
            if fxdata.index.isin(list1).any():
                rate=fxdata.loc[date,currency]
            else:
                rate=prevrate
            usd_prices.loc[date,stocks['Ticker'].values]=usd_prices.loc[date,stocks['Ticker'].values]/rate
            prevrate=rate
            
    
    return usd_prices
            
if __name__ == '__main__':
    
    #read the currency for all tickers from fx2.csv file
    fx=pd.read_csv('fx2.csv')

    #tickers2=['NESN.VX', 'NOVN.VX', 'HSBA.L', 'ROG.VX', 'FP.PA', 'RDSB`.L', 'BP.L', 'SAP.SW', 'BATS.L', 'SIE.DE', 'GSK.L', 'ZEG.HM', 'SNW.DE', 'ALV.DE', 'MC.PA', 'DGE.L', 'UNIA.AS', 'ASMF.DU', 'BAYN.DE', 'NOVO-B.CO', 'BASA.DE', 'SAN.MC', 'ABI.BR', 'AIR.PA', 'ULVR.L', 'BNP.PA', 'RB.L', 'VOD.L', 'UBSG.VX', 'OR.PA', 'PRU.L', 'DAI.DE', 'LLD2.DE', 'RIO.L', 'S7E.F', 'SQU.MU', 'INGA.AS', 'AI.PA', 'DTE.DU', 'CS.PA', 'BN.PA', 'GLEN.L', 'ABBN.VX', 'ENI.MI', 'ZURN.VX', 'CFR.VX', 'BLT.L', 'SAF.PA', 'IBE.MC', 'ADS.DE', 'SU.PA', 'BBVA.MC', 'PHIA.AS', 'ENEL.MI', 'TEF.MC', 'PPX.SG', 'AMS.MC', 'BARC.L', 'LIN.DE', 'ISP.MI', 'IMB.L', 'CSGN.VX', 'NNGE.BE', 'ITX.MC', 'DPW.DE', 'CPG.L', 'ORA.MI', 'RI.PA', 'NDA-SEK.ST', 'GLE.PA', 'TSCO.L', 'MUV2.DE', 'UCG.MI', 'EI.PA', 'FRE.F', 'BMW.DU', 'AD.AS', 'UL.PA', 'NOKIA.HE', 'VOW.DE', 'EQNR.OL', 'GSZ.PA', 'IFX.DE', 'SREN.VX', 'CRG.IR', 'VOLV-B.ST', 'BA.L', 'VVU.HA', 'BT-A.L', 'AV.L', 'DB1.DE', 'SAMPO.HE', 'RR.L', 'EOAN.DE', 'REP.MC', 'WDI.EX', 'ANN.SG', 'HEIA.AS', 'LONN.VX', 'REL.L', 'LHN.VX', 'AKZA.AS', 'STAN.L', 'CON.DE', 'G.MI', 'SGO.PA', 'J2B.F', 'ERIC-B.ST', 'ML.PA', 'HEN3.DE', 'CAP.PA', 'SWED-A.ST', 'DBK.DE', 'KCR1V.HE', 'REN.AS', 'DSY.PA', 'FME.DE', '0WP.BE', 'SKY.L', 'DSN.F', 'INVE-B.ST', 'ASSA-B.ST', 'LGEN.L', 'LR.PA', 'SAND.ST', 'KBC.BR', 'DNB.OL', 'NGLB.MU', 'UPM.HE', 'KPN.AS', 'GIVN.VX', 'ISC1.F', 'SHB-A.ST', 'WOSB.MU', 'RMS.PA', 'FERG.L', 'SIKA.VX', 'LSE.L', 'ATCO-A.ST', 'FCA.MI', 'RNL.HM', 'DSV.CO', 'KRZ.BE', 'SEB-A.ST', 'XCA.HA', '1COV.DE', 'SSE.L', 'PGHN.VX', 'ESSITY-B.ST', 'GEBN.VX', 'DWNI.SW', 'UG.PA', 'SN.L', 'HEXA-B.ST', 'CABK.MC', 'RACE.MI', 'AHT.L', 'NN.AS', 'HO.PA', 'PUB.PA', 'SGSG.VX', 'MRK.F', 'MRO.L', 'UHR.VX', 'TEL.OL', 'FTI.PA', 'RWE.DE', 'VWS.SW', 'AENA.MC', 'STM.MI', 'CARL-B.CO', 'TELIA.ST', 'INF.L', 'HM-B.ST', 'ABN.AS', 'COLO-B.CO', 'STA.L', 'EN.PA', 'BAER.VX', 'SLHN.VX', 'BRBY.L', 'IHG.L', 'EBS.VI', 'LUX.MI', 'VIE.PA', 'HEI.DE', 'SCMN.VX', 'CNHI.MI', 'III.L', 'NZYM-B.CO', 'UCB.BR', 'MTX.DE', 'OXBEI.SW', 'UMI.BR', 'RBS.L', 'TKA.DE', 'ITRK.L', 'OCI1.BE', 'OXSY1.SW', 'AGN.AS', 'VSA.F', 'ABF.L', 'NESTE.HE', 'CCL.L', 'ORSTED.CO', 'FGR.PA', 'FORTUM.HE', 'CAN.L', 'ATO.PA', 'ATL.MI', 'AC.PA', 'TEMN.SW', 'AGS.BR', 'TEP.PA', 'SWMA.ST', 'BNZL.L', 'CA.PA', 'ADEN.VX', 'FER.MC', 'GE9.DU', 'SOON.VX', 'EDP.LS', 'CHR.CO', 'MNDI.L', 'NXT.L', 'PSN.L', 'GALP.LS', 'SJ7.F', 'SGE.L', 'SOLB.BR', 'NTGY.MC', 'SRG.MI', 'QSV.MU', 'PAH3.DE', 'MAERSK-B.CO', 'STERV.HE', 'WTBF.EX', 'WRT1V.HE', 'CBK.DE', 'SCHP.VX', 'PSON.L', 'JMT1.BE', 'IAG.L', 'BNR.DE', 'GFC.PA', 'REE.MC', 'RY4C.IR', 'SK3.IR', 'MHG.OL', 'KNIN.VX', 'LISN.SW', 'CRDA.L', 'STMN.SW', 'KPN.AS', 'QIA.DE', 'V1SF.EX', 'OMUB.MU', 'ITV.L', 'SAB.BC', 'TUI.L', 'SGRO.L', 'RSA.L', 'OCDO.L', 'ZAL.BE', 'NHY.OL', 'TWW.DU', '4BV.BE', 'LXS.SW', 'ENG.MC', 'OPA.F', 'W7LF.EX', 'KN.PA', 'JE.L', 'BKT.MC', 'VOE.VI', 'ZS3.F', 'GTO.AS', 'KGX.HA', 'GRB.MU', 'VLM.MU', 'IYYA.BE', 'MF.PA', 'TQW.SG', 'JYS1.BE', 'WILB.BE', 'XHSA.DU', 'XNP.F', 'RF.PA', 'XCL1.SG', 'FRA.SW', 'KONN.F', 'WHI.DU', 'MAP.MC', 'SOP.PA', 'VIS.F', 'SBMO.AS', 'WCH.HA', 'MEO.DE', 'KAZ.L']
    
    tickers2=['DRI.DE', 'III.L', 'MAERSK-B.CO', 'A2A.MI', 'AALB.AS', 'ARL.DE', 'ABBN.VX', 'ABN.AS', 'AC.PA', 'ACKB.BR', 'OCI1.BE', 'ADEN.VX', 'ADS.DE', 'ADM.L', 'W7LF.EX', 'AGN.AS', 'AENA.MC', 'AGS.BR', 'AGK.L', 'AD.AS', 'A5G.IR', 'AF.PA', 'AI.PA', 'AIR.PA', 'AKERBP.OL', 'AKZA.AS', 'ALFA.ST', 'ALV.DE', 'ALO.PA', 'ATE.PA', 'ATC.AS', 'ALT.PA', 'AMS.MC', '0MJF.IL', 'AMEAS.HE', 'AMS.SW', 'AEEM.PA', 'ANDR.VI', 'NGLB.MU', 'ABI.BR', 'ANTO.L', 'ISC1.F', 'ARGX.BR', 'V1SF.EX', 'AT1.DE', 'AHT.L', 'ASM.AS', 'ASML.AS', 'ASRNL.AS', 'ASSA-B.ST', 'G.MI', 'ABF.L', 'ZEG.HM', 'ATL.MI', 'ATCO-A.ST', 'ATO.SE', 'NDA.DE', 'AUTO.L', 'AV.L', 'CS.PA', 'BME.L', 'BAB.L', 'BA.L', 'BBY.L', 'BALN.VX', 'BAMI.MI', 'BIRG.IR', 'BKT.MC', 'BARC.L', 'BDEV.L', 'BARN.SW', 'BASE.DE', 'BAYN.DE', 'BBA.L', 'BBVA.MC', 'SAB.MC', 'SAN.MC', 'BESI.AS', 'BEZ.L', 'OXBEI.SW', 'BWY.L', 'BKG.L', 'BLT.L', 'BIC.PA', 'BILL.ST', 'BIM.PA', 'BMW.DE', 'BNP.PA', 'BOL.ST', 'BME.MC', 'BOKA.AS', 'EN.PA', 'BP.L', 'BPE.MI', 'BNR.DE', 'BATS.L', 'BLND.L', 'BVIC.L', 'BT-A.L', 'BTG.L', 'BUCN.SW', 'BNZ.L', 'BRBY.L', '4BV.BE', 'CABK.MC', 'CAP.PA', 'CPI.L', 'CAPC.L', 'CARL-B.CO', 'CCL.L', 'CA.PA', 'CO.PA', 'CAST.ST', 'CLNX.MC', 'CMBN.SW', 'CAN.L', 'CHR.CO', 'CDI.PA', 'CFR.VX', 'CINE.L', 'CLN.VX', 'CBG.L', 'CNHI.VI', 'XNP.F', 'COB.L', 'CCH.L', 'COFB.BR', 'COLO-B.CO', 'CBK.DE', 'CPG.L', 'CON.DE', 'CTEC.L', '1COV.DE', 'COV.PA', 'XCA.HA', 'CSGN.VX', 'CRG.IR', 'CRDA.L', 'CYBG.L', 'DMGT.L', 'DAI.DE', 'DANOY', 'DSN.F', 'AM.PA', 'DSY.PA', 'CPR.MI', 'DCC.L', 'DPH.L', 'DHER.DE', 'DLN.L', 'DBK.DE', 'DB1.DE', 'DPW.DE', 'DTE.DE', 'DWNI.SW', 'DGE.L', 'DLG.L', 'DC.L', 'DKSHF', 'DNB.OL', 'DOM.ST', 'DOKA.SW', 'SMDS.L', 'DSV.CO', 'DUE.DE', 'DUFN.VX', 'EOAN.DE', 'EZJ.L', 'QSV.MU', 'EDF.PA', 'EDPL.LS', 'FGR.PA', 'EENEF', 'ELUX-B.ST', 'EKTA-B.ST', 'ELIS.PA', 'ELISA.HE', 'EMSN.SW', 'ENG.MC', 'ELEC.MC', 'ENEL.MI', 'ENGI.PA', 'ENI.MI', 'EPI-B.ST', 'EQNR.OL', 'ERIC-B.ST', 'EBS.VI', 'EI.PA', 'ESSITY-B.ST', 'COLR.DR', 'RF.PA', 'ERF.PA', 'ENX.PA', 'ETL.PA', 'EVK.DE', 'EXO.MI', 'J2B.F', 'WILB.BE', 'BALD-B.ST', 'EO.PA', 'FERG.L', 'RACE.MI', 'FER.MC', 'FCA.MI', 'ZS3.F', 'FLS.CO', 'FHZN.SW', 'FORTUM.HE', 'FRA.SW', 'FNTN.DE', 'FRE.F', 'FME.DE', 'FRES.L', 'FPE.DE', 'GLPG.AS', 'GALP.LS', 'GAM.SW', 'G1A.DE', 'GEBN.VX', 'GFC.PA', 'GTO.AS', 'GE9.DU', 'GXI.DE', 'GETI-B.ST', 'GET.PA', 'GIVN.VX', 'GJF.OL', 'GL9.IR', 'GSK.L', 'GN.CO', 'GPOR.L', 'GLJ.DE', 'GRF.MC', 'GFS.L', 'GBLB.BR', 'GLE.PA', 'GCV.L', 'LUN.CO', 'HLMA.L', 'HMSO.L', 'HNR1.DE', 'HL.L', 'HAS.L', 'HEI.DE', 'HEIA.AS', 'HEIO.AS', 'HLE.DE', 'HELN.SW', 'HEN3.DE', 'HM-B.ST', 'RMS.PA', 'HEXA-B.ST', 'HSX.L', 'HOT.DE', 'XHSA.DU', 'HWDN.L', 'HSBA.L', 'BOSS.DE', 'HRH1V.HE', 'HUSQ-B.ST', 'IAG.L', 'IBE.MC', 'ICA.ST', 'ICAD.PA', 'IGG.L', 'ILD.PA', 'IMCD.AS', 'NK.PA', 'IMI.L', 'IMB.L', 'INCH.L', 'INDV.L', 'IDEXY', 'INDU-A.ST', 'IFX.DE', 'INF.L', 'INGA.AS', 'ING.PA', 'ISAT.L', 'COL.MC', 'IGY.DE', 'IHG.L', 'ICP.L', 'ITRK.L', 'ISP.MI', 'INTU.L', 'IYYA.BE', 'INVE-B.ST', 'IPN.PA', 'ISS.CO', 'IG.MI', 'ITV.L', 'IWG.L', 'DEC.PA', 'JMT.LS', 'JMT1.BE', 'BAER.VX', 'JUP.L', 'JE.L', 'JYS1.BE', 'SDF.DE', 'KAZ.L', 'KBC.BR', 'PPX.SG', 'KRZ.IR', 'KESKOB.HE', 'KIND-SDB.ST', 'KGF.L', 'KRX.IR', 'KINV-B.ST', 'KGX.HA', 'LI.PA', 'KONN.F', 'KNEBV.HE', 'KCR.HE', 'DSM.AS', 'KPN.AS', 'KNIN.VX', 'ORN.MX', 'LHN.VX', 'MMB.PA', 'LAND.L', 'LXS.SW', 'LEG.DE', 'LGEN.L', 'LR.PA', 'LDO.MI', 'LIN.DE', 'LISN.SW', 'LLOYL.L', 'LOGN.VX', 'LSE.L', 'LONN.VX', 'LOOM-B.ST', 'LHA.DE', 'LUND-B.ST', 'LUPE.ST', 'LUX.MI', 'MA.PA', 'MAN.DE', 'EMG.L', 'MAP.MC', 'MHG.OL', 'MKS.L', 'MDC.L', 'MB.MI', 'MGGT.L', 'MRO.L', 'MRK.F', 'MERL.L', 'MRL.MC', 'MTAGF', 'MTRO.L', 'VLM.MU', 'ML.PA', 'MCRO.L', 'MONC.MI', 'MNDI.L', 'MONY.L', 'MOR.DE', 'MRW.L', 'MTX.DE', 'MUV2.DE', 'NNGE.BE', 'KN.PA', 'GASNF.MC', 'NESTE.HE', 'NESN.VX', 'SMWH.L', 'NXG.L', 'NXT.L', 'NIBE-B.ST', 'NN.AS', 'NOKIA.HE', 'NRE1V.HE', 'NDA-SEK.ST', 'NHY.OL', 'NOVN.VX', 'NOVO-B.CO', 'NZYM-B.CO', 'OELR.SW', 'OCDO.L', 'OMUB.MU', 'OMV.VI', 'ORNBV.HE', 'ORK.OL', 'OPA.F', 'ORSTED.CO', 'OSR.DE', 'PPB.IR', 'PNDORA.CO', 'PARG.SW', 'PGHN.VX', 'PSON.L', 'PNN.L', 'RI.PA', 'PSN.L', 'UG.PA', 'PHIA.AS', 'PHNX.L', 'PIRC.MI', 'POM.PA', 'PTEC.L', 'PAH3.DE', 'PST.MI', 'PSM.DE', 'PROX.BR', 'PRU.L', 'PRY.MI', 'PSPN.SW', 'PUB.PA', 'QIA.DE', 'QLT.L', 'RBI.VI', 'RRS.L', 'RAND.AS', 'RB.L', 'REC.MI', 'REE.MC', 'REL.L', 'RDLSF', 'RCO.PA', 'RNL.HM', 'RTO.L', 'REP.MC', 'RXL.PA', 'RHM.DE', 'RMV.L', 'RIO.L', 'ROG.VX', 'ROCK-B.CO', 'RR.L', 'ROR.L', 'RBS.L', 'RDSA.L', 'RMG.L', 'RBREW.CO', 'RPC.L', 'RSA.L', 'RTL.BR', 'RUBSF', 'RWE.DE', 'RYA.L', 'SAAB-B-.ST', 'SAF.PA', 'SGE.L', 'SBRY.L', 'SGO.PA', 'SPM.MI', 'SAMPO.HE', 'SAND.ST', 'SNW.DE', 'SAP.SW', 'SRT3.DE', 'SBMO.AS', 'SHA.DE', 'SCHA.OL', 'SCHP.VA', 'SU.PA', 'SDR.L', 'SCR.PA', 'SSE.L', 'G24.DE', 'GRB.MU', 'SECU-B.ST', 'SGRO.L', 'SESG.PA', 'SVT.L', 'SGSN.VX', 'S7E.F', 'SIE.DE', 'SGRE.MC', 'SHL.DE', 'LIGHT.AS', 'SIKA.VX', 'WAF.DE', 'XCL1.SG', 'SEB-A.ST', 'SKA-B.ST', 'SKF-B.ST', 'SKY.L', 'SN.L', 'SMN.L', 'SK3.IR', 'SRG.MI', 'SJ7.F', 'SOW.DE', 'SOLB.BR', 'SOON.VX', 'SOP.PA', 'SXS.L', 'SPIE.PA', 'SPX.L', 'SPR.DE', 'SSPG.L', 'STJ.L', 'STAN.L', 'SLA.L', 'STM.MI', 'STERV.HE', 'STB.OL', 'STMN.VX', 'SUBC.OL', 'SEV.PA', 'SRCG.SW', 'SCA-B.ST', 'SHB-A.ST', 'UHR.VX', 'SWED-A.ST', 'SWMA.ST', 'SOBI.ST', 'SLHN.VX', 'SPSN.SW', 'SREN.VX', 'SCMN.VX', 'SYDB.CO', 'OXSY1.SW', 'TEG.DE', 'TATE.L', 'TWW.DU', 'TECN.VX', 'FTI.PA', 'TEL2-B.ST', 'TIT.MI', 'TEF.MC', 'O2D.DE', 'TNET.BR', 'TEL.OL', 'TEP.PA', 'TELIA.ST', 'TEMN.SW', 'TEN.MI', 'TRN.MI', 'TSCO.L', 'TGS.OL', 'HO.PA', 'TKA.DE', 'FP.PA', 'TCAP.L', 'TPK.L', 'TREL-B.ST', 'TRYG.CO', 'TIU.L', 'TQW.SG', 'UBI.MI', 'UBI.PA', 'UBSG.VX', 'UCB.BR', 'UDG.L', 'UMI.BR', 'UCG.MI', 'UNLVF', 'UL', 'UN01.DE', 'UTDI.DE', 'UU.L', 'UMP.HE', 'VSA.F', 'VACN.SW', 'VIE.PA', 'VWS.SW', 'VCT.L', 'VIFN.VX', 'DG.PA', 'VIS.F', 'VVU.HA', 'VOD.L', 'VOE.VI', 'VOW3.DE', 'VOLV-B.ST', 'VONOY', 'VPK.AS', 'WCH.HA', 'WRT1V.HE', 'WEIR.L', 'WNDLF', 'WTBN.MX', 'WIE.VI', 'WDH.CO', 'WHI.DU', 'WDI.EX', 'WOSB.MU', 'WG.L', '0WP.BE', 'YAR.OL', 'ZAL.BE', 'ZURN.VX']
    
    
    random.seed(3)
    lists=random.sample(range(0,len(tickers2)),len(tickers2))
    
    #start date
    start = datetime.datetime(2010, 5, 31)
    
    #end date
    end = datetime.datetime(2018, 5, 31)
    
    datas=pd.DataFrame()
    fxdatas=pd.DataFrame()
    
    volumes=pd.DataFrame()
    
    
    tickers=[tickers2[i] for i in lists]
    
    i=0
    
    #we need to download 500 stocks and the functions fails so I download it in batches of 130
    while i<len(tickers):
        j=130
        if i+130 >= len(tickers):
            j=len(tickers)-i-1
        #download the data
        quotes=download_data(tickers[i:i+j],start, end)
        data=quotes.iloc[:, quotes.columns.get_level_values(0)=='Adj Close']
        volume_data=quotes.iloc[:, quotes.columns.get_level_values(0)=='Volume']
        
        frames=[datas,data]
        vol=[volumes,volume_data]
        volumes=pd.concat(vol, axis=1)

        datas=pd.concat(frames, axis=1)
        i=i+131
        time.sleep(5)
        
    
    datas=datas.sort_index(ascending=[True])
    data=datas
    
    datas = datas.fillna(method='ffill')
    datas.columns=datas.columns.get_level_values(1)
   
    
    volumes=volumes.sort_index(ascending=[True])
    volumes.columns=volumes.columns.get_level_values(1)
     
    
    
    mean_volume=volumes.mean(axis=0)
    
    # we have to clean the data and remove all stocks which have very low liduidity or data issues. 
    # we remove any stocks which does have a volume of more than 1000 over the entire time period
    volumes=volumes.dropna(axis=1,how='all')
    valid_stocks=mean_volume[mean_volume>1000].index.values
    
    datas=datas.loc[:,valid_stocks]
    volumes=volumes.loc[:,valid_stocks]
    datas=datas.loc[:,valid_stocks]
    
    
    new_volume=volumes[volumes!=0]
    cvolume=new_volume.count(axis=0)
    
    #we also filter out any stocks which dont have a volume data available for atleast 1500 days (out of aroudn 2100 days)
    valid_stocks2=cvolume[cvolume>1500].index.values
    
    
    datas=datas.loc[:,valid_stocks2]
    volumes=volumes.loc[:,valid_stocks2]
    datas=datas.loc[:,valid_stocks2]
    
    datas.to_csv('prices.csv')
    volumes.to_csv('volumes.csv')
    
    tickers3=['CHF=X', 'DKK=X', 'EUR=X', 'GBP=X', 'NOK=X', 'SEK=X']
    
    i=0
    #download the fx rate for the time period
    while i<len(tickers3):
        j=130
        if i+130 >= len(tickers3):
            j=len(tickers3)-i
        #download the data
        quotes=download_data(tickers3[i:i+j],start, end)
        
        fxdata=quotes.iloc[:, quotes.columns.get_level_values(0)=='Adj Close']
        
        frames=[fxdatas,fxdata]
        fxdatas=pd.concat(frames, axis=1)
        i=i+131
        time.sleep(5)
    
    
    fxdatas=fxdatas.sort_index(ascending=[True])
    fxdatas = fxdatas.fillna(method='ffill')
    fxdatas = fxdatas.fillna(method='bfill')
    fxdatas.columns=fxdatas.columns.get_level_values(1)
  
    fxdatas.to_csv('fxdata.csv')
    #GBP stocks are expressed in pence and hence we multiply the fx rate by 100
    fxdatas['GBP=X']=fxdatas['GBP=X']*100
    
    lists=datas.columns.values
    
    fx=fx[fx['Ticker'].isin(lists)]
    
    
    usd_prices=convert_to_usd(datas, fx, tickers3, fxdatas)
    
    usd_prices.to_csv('USDPrices.csv')
    
    
    prices=usd_prices
    
   
    prices=prices.sort_index(ascending=[True])
    
    prices = prices.fillna(method='ffill')
    
    
    volumes=volumes.sort_index(ascending=[True])
    volumes = volumes.fillna(method='ffill')
   
    
    returns=pd.DataFrame(index=prices.index)
    
    #calculate daily returns
    returns=prices.iloc[:,:].pct_change(1)

    returns=returns.iloc[1:,:]

    returns.to_csv('returns.csv')
    
    #calculate alphas
    alphas=calc_alphas2(returns,volumes, prices)
    
    alphas.to_csv('alphas.csv')
    
    alphas=alphas.dropna(how='all')
    
    counter=0    
    
    
    
    #decay alphas over last 4 days
    alphas=decay_alpha(alphas,4)
    
    count=0
    
    new_alphas=alphas
    
    prevpos=pd.Series(np.zeros(len(new_alphas.columns)), index=new_alphas.columns)
    
    portfolio=pd.Series()
    
    prevdate=new_alphas.index[0]
    
    pos_trade=pd.DataFrame()
    
    #set maximum weights of alpha as 10% of prtolfio weight
    limit=0.1
    
    
    final_alphas=pd.DataFrame(index=new_alphas.index, columns=new_alphas.columns)
    
    invest=pd.DataFrame(index=new_alphas.index, columns=new_alphas.columns)
    
    prevyear=new_alphas.index[0].year
    
    offset=252
    
    # invest 20,000$ per stock
    unit=20000
    
    calc=1
    
    starts=(counter)*offset
    ends=(counter+1)*offset
    
    #calculate the clusters for first year       
    clusters=create_clusters(prices,volumes,counter)
    
    #iterate through each date 
    for date in new_alphas.index:
        
        if date== new_alphas.index[ends]:
            counter=counter+1
        
        #calculate clusters of stocks based on the preceding 1 year volatility, returns and liquity (shares_outstading*volume)
        if (date==new_alphas.index[0]) or (date== new_alphas.index[ends]):
            
            
            
            #select the preceding 1 year prices and volumes data to calculates clusters
            starts=(counter)*offset
            ends=(counter+1)*offset
            
            print(date)
            
            if (starts > (len(new_alphas.index)-1)):
                
                if new_alphas.index[(counter-1)*offset] <= new_alphas.index[-1]:
                    calc=0
                
                else:
                    break
            
            if ends> (len(new_alphas.index)-1):
                ends=len(new_alphas.index)-1
            
            if calc==1:
                prevclusters=clusters            
                #calculates clusters at 1 year interval
                clusters=create_clusters(prices.iloc[starts:ends,:], volumes.iloc[starts:ends,:], counter)
                cl='cluster'+str(counter)+'.csv'
                clusters.to_csv(cl)
                
        if counter==0:
            prevclusters=clusters
        
        
        if counter>0:
            
            weights=pd.Series()
            investment=pd.Series(index=new_alphas.columns)
            
            
            for cluster in range(0,5):
                
                #iterate through the stocks in each of the 5 clusters and create a cluster neutral portfolio (long short withn cluster based on alphas)
                stocks=prevclusters[prevclusters['Cluster']==cluster].index.values
                
                
                if len(stocks)>1:
                    
                    #normalize the alphas
                    investment.loc[stocks]=unit*len(stocks)
                    sub_alphas=new_alphas.loc[date,stocks]
                
                    alphas_mean=sub_alphas.mean()
                    alphas_normalized=sub_alphas.subtract(alphas_mean)
                
                    #within the cluster, calculate long and shorts so that the total weight is 0 (long total=1 and short total=-1)
                    row=alphas_normalized
        
                    longs=row[row>0]
                    sum1=longs.sum()
        
                    longs=longs/sum1
        
                
                    shorts=row[row<0]
                    sum2=-1*shorts.sum()
        
                    shorts=shorts/sum2
                
                    longs=convert_weights(longs,limit)
                    shorts=convert_weights(shorts,limit)
        
                
                    weights=weights.append(longs)
                    weights=weights.append(shorts)
            
            
            
            final_alphas.loc[date,:]=weights 
            
            
            invest.loc[date,:]=investment
            
            
                
        prevyear=date.year
        
    #save the final alphas wich are the weights of each stock within its clusters
    final_alphas.to_csv('Final_Alphas.csv')
    
    #save the dataframe with total $ investment for each cluster
    invest.to_csv('Invest.csv')
    
    #calculate the number of shares for each stock based on the alpha weights and prices, over the total time period
    pos_trade=final_alphas*invest/prices
    
    
    new_prices=prices.shift(1)
    
    ols_pos_trade=pos_trade
    
    pos_trade=pos_trade.shift(1)
    
    #calculate the daily P&L of the holdings as (today's price-yesterday's price)* yesterday's position
    port1=(prices-new_prices)*pos_trade
    port2=port1.sum(axis=1)
    
    #calculate the cumulative P&L of our strategy as series
    portfolio=port2.cumsum()
    port2.to_csv('port2.csv')
    
    
    portfolio=portfolio.dropna()
    pos_trade.to_csv('PositionHoldings.csv')
    
    portfolio=portfolio.iloc[300:]
    
    portfolio=portfolio+1000000
    
    
    portfolio.to_csv('FinalPortfolio.csv')
    
    plt.figure(3)
    
    #plot the portolfio performance
    ax=portfolio.plot(legend=False, title="Performance of our Portfolio")
    ax.set_xlabel('Dates')
    ax.set_ylabel('Value')
    
    #calculate the risk & return metrics for our portfolio
    profile=riskreturn_profile(portfolio, pos_trade, prices)
                    
    profile.to_csv('Profile.csv')                    
    
        
    
    