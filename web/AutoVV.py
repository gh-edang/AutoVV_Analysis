import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import os
import sys
from web.plateformating import DF96well,createHeatMap,quadrants384_to_96,DF384well
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import statistics
import openpyxl
from datetime import datetime
import scipy.stats
matplotlib.use('TkAgg',force=True)

now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H%M%S")
# Get the base path for your files
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Define the paths for the sample and standard files 
RESULTS_PATH_SAMPLE = r"C:\Users\guardant\Documents\AutoVV\Sample_Location\Sample_fileLocation.txt"
RESULTS_PATH_STD = r"C:\Users\guardant\Documents\AutoVV\STD_Location\STD_fileLocation.txt"
RESULTS_PATH_BACKUPS = r"S:\OncEngDB\AutoVV Spark Output Files\Backups"
# use this line when packaging for pyinstaller
#RESULTS_PATH_LOCAL = os.path.join(base_path, 'web', 'img')
# use this line when running the eel package
RESULTS_PATH_LOCAL = "web/img"
# Create the img folder if it doesn't exist
if not os.path.exists(RESULTS_PATH_LOCAL):
    os.makedirs(RESULTS_PATH_LOCAL)

#defining the standard concentrations for the standard curve
standard_conc_low =[6,12,18,24,30,36,42,48,6,12,18,24,30,36,42,48,55,55]
standard_conc_mid=[49,49,55,59,63,67,71,75,79,83,55,59,63,67,71,75,79,83] 
standard_conc_high=[100,107,114,121,128,135,142,149,100,107,114,121,128,135,142,149]

#defining the expected volume for each method and plate and the range of the plate for heatmap
dict_methods_plate_volume ={
    "BE_PLC": [75,68,83],
    "BE_UTE": [55,47,63],
    "MBDS_PLC (dilution)": [10,3,15],
    "MBDS_PLC (sample)": [64,59,69],
    "MBDS_UTE": [127,117,131],
    "MBDS_CAR":[9,4,12],
    "MBDC_PLC":[32.5,29,36],
    "MBDC_DSB":[55,49,62],
    "LP_ERM": [16,13,19],
    "LP_LPA": [11,3,13],
    "LP_LOM": [15,0,15],
    "LP_LGM": [34,30,38],
    "LP_DSB": [69,56,82],
    "LP_LMO": [77.5,70,86],
    "LP_LMR": [77.5,70,86],
    "LPC_DSB": [45,38,53],
    "LPC_PLC1": [26,23,29],
    "LPC_PLC2": [44,40,48],
    "ENS_PLC1": [18,13,21],
    "ENS_PLC2": [36,33,39],
    "ENS_FHB": [20,5,25],
    "ENS_GBB": [28,23,33],
    "ENS_EBB": [28,23,33],
    "ENW_NOH": [120,115,125],
    "ENW_TRS": [44.5,40,50],
    "ENW_IO_": [83.5,76,91],
    "ENW_EMM": [8,0,12],
    "ENC_DSB": [70,56,85],
    "ENT_PLC": [50,40,60]
} 

def get_Excel_File(initial_file_path):
    sample_path = os.path.split(initial_file_path) 
    file_path = sample_path[0][:-3] + "xlsx"
    files = os.listdir(file_path)
    final_sample_path = os.path.join(file_path,files[0])
    return final_sample_path

def reformat_STD_map_list(start1,end1,start2,end2,df):
    formatted_df = df.drop(range(start1,end1))
    formatted_df = formatted_df.drop(range(start2,end2))
    formatted_df = formatted_df.reset_index(drop = True)
    for i in formatted_df.columns:
        try:
            formatted_df[[i]] = formatted_df[[i]].astype(float).astype(int)
        except:
            pass
    formatted_df.columns = formatted_df.iloc[0]
    formatted_df = formatted_df.drop([0])
    formatted_df.set_index("<>",inplace = True)
    formatted_df.apply(pd.to_numeric)
    return formatted_df

def reformat_STD_list_to_DF(start1,end1,start2,end2,df):
    formatted_df = df.drop(range(start1,end1))
    formatted_df = formatted_df.drop(range(start2,end2))
    formatted_df = formatted_df["Unnamed: 1"]
    print(formatted_df)
    formatted_df = DF96well(formatted_df)
    print(formatted_df)
    return(formatted_df)

def average_std_list_values(list_1,list_2):
    avg_list_std = []
    for i in range(0, len(list_1)):
        avg_list_std.append((list_1[i] + list_2[i])/2)
    return(avg_list_std)
    

def autoVV_Analysis():
    #grabbing txt file from c# for file paths 
    sample_file_path = open(RESULTS_PATH_SAMPLE, "r")
    std_file_path = open(RESULTS_PATH_STD, "r")
    lst_std_path = []
    lst_sample_info = []
    for x in sample_file_path:
        x =x[:-1]
        lst_sample_info.append(x)
    for x in std_file_path:
        x = x[:-1]
        lst_std_path.append(x)

    print(lst_sample_info)
    print(lst_std_path)
    sample_path = get_Excel_File(lst_sample_info[0])
    std_path = get_Excel_File(lst_std_path[0])
    method_plate = lst_sample_info[1]+ "_"+ lst_sample_info[2]
    bln_beadsub= lst_sample_info[3]
    print(method_plate)
    print(sample_path)
    print(std_path)
    folder_name = dt_string + "_"+ method_plate
    save_path = os.path.join(RESULTS_PATH_BACKUPS,folder_name)

    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    #formmating the STD file into low mid and high
    std_df1 = pd.read_excel(std_path, header = None)
    std_df = pd.read_excel(std_path)
    
    instrument_SN = std_df1.at[2,4]
    instrument_SN = instrument_SN.replace('Serial number: ',"")
    date_time1 = std_df1.loc[43,4]
    print("This is Tecan Spark SN: ",instrument_SN)
    
    try:    
        std_low = reformat_STD_map_list(0,45,54,129,std_df)
        std_mid = reformat_STD_map_list(0,79,88,129,std_df)
        std_high = reformat_STD_map_list(0,113,122,129,std_df)
        
        low_coln1 = std_low[1].tolist()
        low_coln2 = std_low[2].tolist()
        low_coln3 = std_low[3].tolist()
        low_coln4 = std_low[4].tolist()
        mid_coln1 = std_mid[1].tolist()
        mid_coln2 = std_mid[2].tolist()
        mid_coln3 = std_mid[3].tolist()
        mid_coln4 = std_mid[4].tolist()

        high_coln1 = std_high[5].tolist()
        high_coln2 = std_high[6].tolist()
        
    except KeyError:
        std_low = reformat_STD_list_to_DF(0,56,152,403,std_df)
        std_mid = reformat_STD_list_to_DF(0,178,274,403,std_df)
        std_high = reformat_STD_list_to_DF(0,300,396,403,std_df)

        low_coln1 = std_low["1"].tolist()
        low_coln2 = std_low["2"].tolist()
        low_coln3 = std_low["3"].tolist()
        low_coln4 = std_low["4"].tolist()

        mid_coln1 = std_mid["1"].tolist()
        mid_coln2 = std_mid["2"].tolist()
        mid_coln3 = std_mid["3"].tolist()
        mid_coln4 = std_mid["4"].tolist()

        high_coln1 = std_high["5"].tolist()
        high_coln2 = std_high["6"].tolist()

    low_list_std= low_coln1+low_coln2
    low_list_std.append((low_coln3[0]))
    low_list_std.append(low_coln4[0])
    print(low_list_std)

    mid_list_std=[]
    mid_list_std.append(mid_coln1[7]) 
    mid_list_std.append(mid_coln2[7]) 
    mid_list_std= mid_list_std + mid_coln3+mid_coln4
    print(mid_list_std)

    high_list_std = high_coln1+high_coln2
    print(high_list_std)
    list_sample = dict_methods_plate_volume[method_plate]
    vol_expected = float(list_sample[0])
    vol_min = float(list_sample[1])
    vol_max = float(list_sample[2])
    str_vol_range = str(vol_min) + " - " + str(vol_max)
    if vol_expected <50:
        expected_volume_std = standard_conc_low
        generated_volume_std = low_list_std
    elif vol_expected >= 50 and vol_expected < 100:
        expected_volume_std = standard_conc_mid
        generated_volume_std = mid_list_std
    elif vol_expected >= 100:
        expected_volume_std = standard_conc_high
        generated_volume_std = high_list_std
    print("standard RFUs: ",generated_volume_std)
    print("standard volume Expected: ",expected_volume_std)
    
    #std_formatted_96well = DF96well(new_std_list) # just used for printing into 96 well heat map
    #calculating the coefficients for the standard curve and the r2 value based off of the standard values
    coeffs = np.polyfit(expected_volume_std,generated_volume_std, 2)
    #print(coeffs)
    std_equation = "y = " + str(round(coeffs[0],2))+"x**2 + "+str(round(coeffs[1],2)) +"x + " + str(round(coeffs[2],2))
    print(std_equation)
    r_squared = np.poly1d(coeffs)
    coeff_of_dermination = round(r2_score(generated_volume_std, r_squared(expected_volume_std)),2)

    # Train polynomial regression model on the whole dataset
    pr = PolynomialFeatures(degree = 2)
    X_poly = pr.fit_transform(np.array(generated_volume_std).reshape(-1, 1))
    lr_2 = LinearRegression()
    lr_2.fit(X_poly, expected_volume_std)
    y_pred_poly = lr_2.predict(X_poly) 
    print(y_pred_poly)

    # Visualising the Polynomial Regression results
    plt.figure(figsize=(8, 8))
    plt.scatter(generated_volume_std,expected_volume_std)
    plt.plot(generated_volume_std, lr_2.predict(X_poly),color = 'firebrick')
    plt.title("Standard Curve")
    plt.ylabel("Concentration (uL)")
    plt.xlabel("Raw Data (RFU)")

    # Adding in the R2 value and the equation of the line
    plt.text(0.05 , 0.86, 'R2 = {}'.format(coeff_of_dermination), fontsize=14,transform=plt.gca().transAxes)
    plt.text(0.05 , 0.92, std_equation, fontsize=14,transform=plt.gca().transAxes)
    finalname = os.path.join(RESULTS_PATH_LOCAL, "standard_curve.png")
    # finalname = RESULTS_PATH_LOCAL + "standard_curve.png"

    #print(finalname)
    plt.savefig(finalname)

    #formatting the sample data into a list so that we can parse it (96 well format to long list)
    sample_df = pd.read_excel(sample_path)
    try:
        sample_df = reformat_STD_map_list(0,43,52,59,sample_df)
    except KeyError:
        sample_df = reformat_STD_list_to_DF(0,54,152,156,sample_df)
    print(sample_df)
    sample_list = sample_df.values.ravel(order="F").tolist()
    print(sample_list)
    volume_calculated = []
    expected_blanks = []
    bead_samples = []
    #calculating the volume based off of the standard curve
    for x,value in enumerate(sample_list):
        try:
            int_value = int(value)
        except ValueError:
            int_value = 0 
        if bln_beadsub == "True":
            if x >=41 and x <= 56:
                expected_blanks.append(round(lr_2.predict(pr.fit_transform([[int_value]]))[0],2))
            else:
                bead_samples.append(round(lr_2.predict(pr.fit_transform([[int_value]]))[0],2))
        volume_calculated.append(round(lr_2.predict(pr.fit_transform([[int_value]]))[0],2))

    if bln_beadsub == "True":
        avg_blanks = round(statistics.fmean(expected_blanks),2)
        avg_samples = round(statistics.fmean(bead_samples),2)
        volume_calculated= [x - (avg_samples-avg_blanks) for x in volume_calculated]

    # for value in sample_list:
    #     try:
    #         int_value = int(value)
    #         volume_calculated.append(round(lr_2.predict(pr.fit_transform([[int_value]]))[0],2))
    #     except ValueError:
    #         int_value = 0 
    #         volume_calculated.append(0)

    final_volume = DF96well(volume_calculated)
    std_formatted_96well = DF96well(generated_volume_std)
    print(final_volume)
    #generating heat maps for the volume data and the standard raw data
    createHeatMap(final_volume,15,7,"sample_heatmap.png",1,RESULTS_PATH_LOCAL,vol_max,vol_min,"manual")
    createHeatMap(std_formatted_96well,15,7,"STD_plate_map.png",1,RESULTS_PATH_LOCAL,vol_max,vol_min,"auto")
    results_dict = {}

    results_dict= findingStatistics(volume_calculated,vol_expected,vol_min,vol_max,method_plate)
    
    return(results_dict,vol_expected,method_plate,base_path,str_vol_range,instrument_SN,date_time1)



def findingStatistics(volume,targetVolume,vol_min,vol_max,method_plate):
    filter_volume = []
    absolute_error = []
    out_of_range = []       

    for x,y in enumerate(volume):
        str_y = str(y)

        if str_y =="NaN" or str_y =="nan" or y ==0:
            pass
        else:
            if method_plate == "LP_LPA":
                if x <33 or x>65:
                    filter_volume.append(float(str_y))
                    volume_error = abs((y-targetVolume)/targetVolume)
                    absolute_error.append(volume_error)
            else:
                volume_error = abs((y-targetVolume)/targetVolume)
                absolute_error.append(volume_error)
                filter_volume.append(float(str_y))
                if y>vol_max or y<vol_min:
                    out_of_range.append(y)
        
    print("Volume: ",filter_volume)

    int_len_out_of_range = len(out_of_range)
    print("Out of range: ",int_len_out_of_range)
    print(absolute_error)
    abs_error=(statistics.mean(absolute_error))*100
    Results_dict = {
        "sample_avg": round(statistics.fmean(filter_volume),2),
        "sample_std": round(statistics.stdev(filter_volume),2),
        "sample_median" : round(statistics.median(filter_volume),2),
        "sample_min" : round(min(filter_volume),2),
        "sample_max" : round(max(filter_volume),2),
        "sample_range" : round(max(filter_volume)-min(filter_volume),2),
    }
    cv =(abs(Results_dict["sample_std"]/Results_dict["sample_avg"])*100)
    # abs_error =(abs((Results_dict["sample_avg"]-float(targetVolume))/float(targetVolume))*100)
    # if abs_error<5.0 and cv <5.0:
    #     sample_passfail = "PASS"
    # else:
    #     sample_passfail = "FAIL"

    if cv <5.0 and int_len_out_of_range==0:
        sample_passfail = "PASS"
    elif cv>5.0 and int_len_out_of_range==0:
        sample_passfail = "PASS - MONITOR"
    elif int_len_out_of_range>10:
        sample_passfail = "FAIL"
    elif int_len_out_of_range<10 and int_len_out_of_range>0:
        sorted_volume = sorted(volume)
        #removing outliers 
        filtered_volume = sorted_volume[5:91]
        filtered_stdev =  round(statistics.stdev(filtered_volume),2)
        filtered_median = round(statistics.median(filtered_volume),2)
        #95 CI
        filtered_CI_95_high = filtered_median+ (1.96*(filtered_stdev))
        filtered_CI_95_low = filtered_median- (1.96*(filtered_stdev))
        print(filtered_CI_95_high)
        print(filtered_CI_95_low)
        #99 CI
        filtered_CI_99_high = filtered_median+ (2.57*(filtered_stdev))
        filtered_CI_99_low = filtered_median- (2.57*(filtered_stdev))
        print(filtered_CI_99_high)
        print(filtered_CI_99_low)
        count_95 = 0
        count_99 = 0
        for x in filtered_volume:
            if x>filtered_CI_95_high or x<filtered_CI_95_low:
                count_95 = count_95 +1
            elif x>filtered_CI_99_high or x<filtered_CI_99_low:
                count_99 = count_99 +1
        print(count_95)
        print(count_99)
        if count_95>2:
            sample_passfail = "FAIL"
        else:
            sample_passfail = "PASS"


    sample_abserror =str(round(abs_error,2)) + "%"
    sample_cv = str(round(cv,2)) + "%"
    Results_dict["abs_error"] = sample_abserror
    Results_dict["sample_cv"]= sample_cv
    Results_dict["pass_fail"]= sample_passfail
    print(Results_dict)

    return Results_dict

# results = {}
# results,vol_expected,method_plate= autoVV_Analysis()
