import eel
from datetime import datetime
from web.AutoVV import autoVV_Analysis
import os
eel.init('web')

@eel.expose
def data_exported():
        results = {}
        results,target_volume,plate_name,savepath,str_vol_range = autoVV_Analysis()
        now = datetime.now()
        dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
        date_time = now.strftime("%m/%d/%Y%H%M%S")
        savepath_sample_heatmap = os.path.join(savepath,"sample_heatmap.png")
        savepath_std_curve = os.path.join(savepath,"standard_curve.png")
        savepath_std_heatmap = os.path.join(savepath,"STD_plate_map.png")
        template_vars96 = {
                "passfail": results["pass_fail"],
                "hidden_show": "style =visibility: hidden;",
                "date": dt_string,
                "target_volume": target_volume,
                "target_range": str_vol_range,
                "method_plate_name": plate_name,
                "path_sample_heatmap": savepath_sample_heatmap,
                "path_std_plot": savepath_std_curve,
                "path_std_heatmap": savepath_std_heatmap,
                "save_file_name": plate_name+"_results_"+date_time+".pdf",
                # "path_sample_heatmap": "img/sample_heatmap.png",
                # "path_std_plot": "./standard_curve.png",
                # "path_std_heatmap": "./STD_plate_map.png",
                "value1": results["sample_avg"],
                "value2": results["sample_std"], 
                "value3": results["sample_cv"], 
                "value4": results["sample_median"], 
                "value5": results["sample_min"], 
                "value6": results["sample_max"], 
                "value7": results["sample_range"], 
                "value8": results["abs_error"], 
                }
        return template_vars96


eel.start('results.html', size=(1100,1000))    