import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

#================================================================================================
# Creating a 96 well plate format into a dataframe to make a heat map afterwards
#================================================================================================
def DF96well(column):
    column1 = []
    column2 = []
    column3 = []
    column4 = []
    column5 = []
    column6 = []
    column7 = []
    column8 = []
    column9 = []
    column10 = []
    column11 = []
    column12 = []

    numSamples = len(column)

    dataFormat = pd.DataFrame(np.zeros((8, 12)), index = ["A","B","C","D","E","F","G","H"])
    dataFormat.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    for x,y in zip(range(numSamples),column):
        round(y,3)
        if x <= 7:
            column1.append(y)
        if x >7 and x<=15:
            column2.append(y)
        if x>15 and x<=23:
            column3.append(y)
        if x>23 and x<=31:
            column4.append(y)
        if x>31 and x<=39:
            column5.append(y)
        if x>39 and x<=47:
            column6.append(y)
        if x>47 and x<=55:
            column7.append(y)
        if x>55 and x<=63:
            column8.append(y)
        if x>63 and x<=71:
            column9.append(y)
        if x>71 and x<=79:
            column10.append(y)
        if x>79 and x<=87:
            column11.append(y)
        if x>87 and x<=95:
            column12.append(y)
    

    columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10,column11, column12]

    for i, column in enumerate(columns):
        if len(column) != 8:
            dataFormat[str(i+ 1)] = (column) + (np.zeros((1, 8 - len(column))) * np.nan).ravel().tolist()                                 
        else:
            dataFormat[str(i+ 1)] = column
    return(dataFormat)

#================================================================================================
# Creating a 384 well plate format into a dataframe to make a heat map afterwards
#================================================================================================

def DF384well(column):
    column1 = []
    column2 = []
    column3 = []
    column4 = []
    column5 = []
    column6 = []
    column7 = []
    column8 = []
    column9 = []
    column10 = []
    column11 = []
    column12 = []
    column13 = []
    column14 = []
    column15 = []
    column16 = []
    column17 = []
    column18 = []
    column19 = []
    column20 = []
    column21 = []
    column22 = []
    column23 = []
    column24 = []

    numSamples = len(column)
    dataFormat = pd.DataFrame(np.zeros((16, 24)), index = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"])
    dataFormat.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12','13','14','15','16','17','18','19','20','21','22','23','24']

    #i feel like theres a dynamic way to iterate through this but im not sure
    for x,y in zip(range(numSamples),column):
        round(y,3)
        if x <= 15:
            column1.append(y)  
        if x >15 and x<=31:
            column2.append(y)
        if x>31 and x<=47:
            column3.append(y)
        if x>47 and x<=63:
            column4.append(y)
        if x>63 and x<=79:
            column5.append(y)
        if x>79 and x<=95:
            column6.append(y)
        if x>95 and x<=111:
            column7.append(y)
        if x>111 and x<=127:
            column8.append(y)
        if x>127 and x<=143:
            column9.append(y)
        if x>143 and x<=159:
            column10.append(y)
        if x>159 and x<=175:
            column11.append(y)
        if x>175 and x<=191:
            column12.append(y)
        if x >191 and x<=207:
            column13.append(y)
        if x>207 and x<=223:
            column14.append(y)
        if x>223 and x<=239:
            column15.append(y)
        if x>239 and x<=255:
            column16.append(y)
        if x>255 and x<=271:
            column17.append(y)
        if x>271 and x<=287:
            column18.append(y)
        if x>287 and x<=303:
            column19.append(y)
        if x>303 and x<=319:
            column20.append(y)
        if x>319 and x<=335:
            column21.append(y)
        if x>335 and x<=351:
            column22.append(y)
        if x>351 and x<=367:
            column23.append(y)
        if x>367 and x<=383:
            column24.append(y)

    columns = [column1, column2, column3, column4, column5, column6, column7, column8, column9, column10,column11, column12,column13, column14, column15, column16, column17, column18, column19, column20, column21, column22,column23, column24]

    for i, column in enumerate(columns):
        if len(column) != 16:
            dataFormat[str(i+ 1)] = (column) + (np.zeros((1, 16 - len(column))) * np.nan).ravel().tolist()                                 
        else:
            dataFormat[str(i+ 1)] = column
    return(dataFormat)

def quadrants384_to_96(data):
    quad1 = []
    for row in np.arange(0, 16, 2):
        for col in np.arange(0, 23, 2):
            quad1.append(data.values[row, col])


    QUAD1_96 = pd.DataFrame(np.array(quad1).reshape([8, 12]),index = ["A","B","C","D","E","F","G","H"]) # 96
    QUAD1_96.columns = np.linspace(1, 12, 12, dtype = np.int_)

    quad2 = []
    for row in np.arange(0, 16, 2):
        for col in np.arange(1, 24, 2):
            quad2.append(data.values[row, col])


    QUAD2_96 = pd.DataFrame(np.array(quad2).reshape([8, 12]),index = ["A","B","C","D","E","F","G","H"]) # 96
    QUAD2_96.columns = np.linspace(1, 12, 12, dtype = np.int_)

    quad3 = []
    for row in np.arange(1, 17, 2):
        for col in np.arange(0, 23, 2):
            quad3.append(data.values[row, col])


    QUAD3_96 = pd.DataFrame(np.array(quad3).reshape([8, 12]),index = ["A","B","C","D","E","F","G","H"]) # 96
    QUAD3_96.columns = np.linspace(1, 12, 12, dtype = np.int_)

    quad4 = []
    for row in np.arange(1, 17, 2):
        for col in np.arange(1, 24, 2):
            quad4.append(data.values[row, col])


    QUAD4_96 = pd.DataFrame(np.array(quad4).reshape([8, 12]),index = ["A","B","C","D","E","F","G","H"]) # 96
    QUAD4_96.columns = np.linspace(1, 12, 12, dtype = np.int_)

    return QUAD1_96, QUAD2_96, QUAD3_96, QUAD4_96

def createHeatMap(data,length, width, name,decimal,path,vol_max,vol_min,heatmap):

    plt.subplots(figsize=(length,width))
    decimalFormat = "." +str(decimal) + "f"
    if heatmap == "auto":
        heatmap_samp = sns.heatmap(data,annot = True, cmap = "Blues",fmt=decimalFormat, linewidth=.5, annot_kws={"fontsize":11})
    else:
        heatmap_samp = sns.heatmap(data,annot = True, cmap = "Blues",fmt=decimalFormat, linewidth=.5, annot_kws={"fontsize":11},vmax=vol_max, vmin=vol_min)
    heatmap_samp.set(xlabel="", ylabel="")
    heatmap_samp.xaxis.tick_top()
    heatmap_samp.yaxis.tick_left()
    heatmap_samp.tick_params(axis='y', rotation=360)
    savename = os.path.join(path, name)
    plt.savefig(savename)
    plt.clf()
