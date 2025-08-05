import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
from sklearn.linear_model._base import _preprocess_data
import json
# from flask_cors import CORS
from utils import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS
from datetime  import date,time,datetime,timedelta, timezone
from tqdm import tqdm
from fastprogress.fastprogress import master_bar, progress_bar
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm
tqdm.pandas()
import os
import glob
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
# from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
from flask_cors import CORS
import re
import joblib
from sklearn.linear_model._base import _preprocess_data
import json
# from flask_cors import CORS
# from utils import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pytz
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import time
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import threading
# Load the model
# model = pickle.load(open('stacked_Updated_Controllables_UPD_SKLEARN.pkl', 'rb'))
model = joblib.load('stacked_with_scaler_18thJuly_controllables_noncontrollables.pkl')
 
# Initialize Flask application
app = Flask(__name__, static_folder='static')
CORS(app, resources={r"*": {"origins": "*"}})
scheduler = BackgroundScheduler()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  
# # Route to handle home page
# @app.route('/')
# def home():
#     return render_template('index.html')
 
@app.route('/')
@app.route('/factoryView')
@app.route('/cascadeView')
@app.route('/assetView')
@app.route('/simulation-model')

def index():
    global Data_entry_df
    Data_entry_df = pd.DataFrame()
    # # Start the scheduler in a separate thread
    # # threading.Thread(target=start_5min_pv_scheduler).start()
    # thread = threading.Thread(target=start_5min_pv_scheduler)
    # thread.start()
    return send_from_directory(directory='static',path='index.html')
 
# Route to handle home page
 
# # Route to handle home page
# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/api/predict_var_19', methods=['POST'])
def predict():
    # Get JSON data from request
    # input_data = request.get_json()
    input_data = request.json
   
 
    df_input=pd.DataFrame(input_data)
 
    df_input = df_input.pivot_table(columns='SensorName', values='InputValue',index=None).fillna('')
    df_input=df_input.reset_index(drop=True)
    df_input.columns = df_input.columns.rename(None)
            # print(df_input)
 
            # final_features = [np.array(df_input['InputValue'])]
    feature_names=[
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',    
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',    
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV']
   
    df_input=df_input.reindex(columns=feature_names)
    df_input = model.named_steps['scaler'].transform(df_input)
    base_rf =model.named_steps['stacked_regressor'].estimators[0][1]
    xgb_model = model.named_steps['stacked_regressor'].estimators[1][1]
    meta_model = model.named_steps['stacked_regressor'].final_estimator
    # Generate predictions from base models
    preds_base_rf = base_rf.predict(df_input)
    preds_base_xgb = xgb_model.predict(df_input)
    stacked_new_predictions = np.hstack([preds_base_rf.reshape(-1, 1), preds_base_xgb.reshape(-1, 1)])
    prediction= meta_model.predict(stacked_new_predictions)
 
 
    # # Initialize lists to store sensor names and input values
    # sensor_names = []
    # input_values = []
 
    # # Extract sensor names and input values from JSON data
    # for item in input_data:
    #     sensor_names.append(item['SensorName'])
    #     input_values.append(float(item['InputValue']))  # Assuming 'InputValue' is numeric
 
    # Make prediction using the loaded model
    # prediction = model.predict([input_values])
 
    # Prepare response JSON
    response = {'prediction': prediction[0]}
 
    # Return JSON response
    # return jsonify({"prediction":'success'})
 
    return jsonify({"prediction":prediction[0]})

mins_pv_result = None
blob_service_client = None

@app.route('/api/predicted_pv', methods=['GET'])
def predicted_pv():

    global mins_pv_result
            
    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    # KUSTO_CLUSTER = "https://dfazuredataexplorer.westeurope.kusto.windows.net/"

    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"

    # KCSB = KustoConnectionStringBuilder.with_aad_device_authentication(KUSTO_CLUSTER)
    # KCSB.authority_id = AAD_TENANT_ID

    # KUSTO_CLIENT = KustoClient(KCSB)

    # In case you want to authenticate with AAD username and password
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)

    active_tags = [
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2015_PVSM_MASS_EXIT_TEMP',
    'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_MIXER_2_REWORK_COUNT',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_SP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TRM_TOP_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TRM_MIDDLE_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_PT_2003_PVFPLDR_PRESSURE',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_800500066581_RECYCLE_PER',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2005_PVPSM_MASS_EXIT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_HARDNESS_PVSOAP_HARDNESS',
    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP',
    'TSPCAS3.Cascade3.PV_FROM_HMI',
    ]

    # Tag_list = ['Tag_abc', 'Tag_def', 'Tag_dsd', 'tag_asds', 'Tag_asad']

    result_tags_num = ", ".join([f"Tag{i}" for i in range(1, len(active_tags) + 1)])

    # print(f'result_tags_num: {result_tags_num}')

    # Tag_list = ['Tag_abc', 'Tag_def', 'Tag_dsd', 'tag_asds', 'Tag_asad']

    result_tags = ""
    for i, tag in enumerate(active_tags, 1):
        result_tags += f"let Tag{i} = \"{tag}\"; "
   
    # print(f"result_tags :{result_tags}")
    KUSTO_QUERY = f'{result_tags} set notruncation;Common2 | where SiteId == "LLPL" | where TS >= ago(15m) | where TS <= now()  | where Tag in ({result_tags_num})| summarize arg_max(TS, *) by Tag |project Tag, TS=datetime_add("minute", 330, TS), Value| evaluate pivot(Tag, max(Value))'

        
    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])
    tagsWithDefaultValue = {'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': '7.34',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': '30.15',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': '-557.94',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': '15.78',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': '7.41',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': '17.06',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': '4018.86',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': '51.74',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': '7.93',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': '28.32',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': '11.23',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': '18.07', 
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': '102.44' , 
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': '458.41',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': '42.49',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': '45.91',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': '38.21',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': '33.59',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': '45.93' }
    # Check if the DataFrame is empty
    if df.empty:
        print('no data')
        # If the DataFrame is empty, return the tag with a value of 0
        response = jsonify({"runningPv_tags":{}, "predictedPv": [], "missing_tags": [], "runningPv":"0"})
        mins_pv_result = response
        return response
    else:
        df=df.replace(r'^\s*$', np.nan, regex=True)
        df_sorted = df.sort_values(by='TS', ascending=False)
        # single_row_dict123 = df_sorted.to_dict(orient='records')[0]
        # print('single_row_dict123', single_row_dict123)
        df_sorted.replace(np.nan,'', inplace=True)
        # print('df_sorted', df_sorted)
        first_non_null_values = {}

       # Iterate through each column and extract the first non-null, non-empty, and non-NaN value
        for column in df_sorted.columns:
            # Filter out null, empty ('') and NaN values
            valid_values = df_sorted[column][df_sorted[column].notna() & (df_sorted[column] != '')]
            
            # Extract the first valid value if available
            first_non_null_values[column] = valid_values.iloc[0] if not valid_values.empty else None

        # print(first_non_null_values)

        # Create a single-row dataframe from the extracted values
        single_row_df = pd.DataFrame([first_non_null_values])
        
        single_row_df.rename(columns={'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'}, inplace=True)
        # Convert the single-row dataframe to a dictionary
        # Print or use `single_row_df` as needed
        # print('single_row_df', single_row_df)
        batch_tags =['LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
            ]
        single_row_df.replace(np.nan,'', inplace=True)
        # print(single_row_df)
        for col in batch_tags:
                        # print(f'col : {col}')
                        single_row_df[col] = single_row_df[col].apply(lambda x: re.search(r'PV:(\d+)', x).group(1) if re.search(r'PV:(\d+)', x) else '')
        single_row_df[batch_tags] = single_row_df[batch_tags].apply(pd.to_numeric, errors='coerce')
        single_row_df['TS'] = pd.to_datetime(single_row_df['TS'])
        single_row_df = single_row_df.dropna(axis=1, how='all')
        single_row_dict = single_row_df.to_dict(orient='records')[0]
        # Get the default value for the specified column
        runningpv_tag_dict = single_row_dict.get('TSPCAS3.Cascade3.PV_FROM_HMI')
        # Ensure the 'TS' key exists and handle missing key
        ts_value = single_row_dict.get('TS', pd.Timestamp.utcnow()).isoformat()
        # print('single_row_dict', single_row_dict)
        feature_names=[
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',    
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',    
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP',
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV']
        
        # Identify columns that contain only empty strings or NaN values
        empty_columns = [col for col in single_row_df.columns if single_row_df[col].isna().all() or (single_row_df[col] == '').all()]

        # print("Empty or NaN columns:", empty_columns)
        # Verify if all columns are empty or NaN
        if len(empty_columns) == len(single_row_df.columns):
            # print("All columns are empty or contain only NaN values.")
            return jsonify({"runningPv_tags":{}, "predictedPv": [], "missing_tags": [], "runningPv":"0"})
        elif len(empty_columns) > 0:
            # print(f"Some columns are empty or contain only NaN values: {empty_columns}")
            # Remove the empty or NaN columns from `single_row_df`
            single_row_df.drop(columns=empty_columns, inplace=True)
               
        
        # Find missing columns
        missing_columns = [col for col in feature_names if col not in single_row_df.columns]
        # Add missing columns to DataFrame with default values
        for col in missing_columns:
            if col in tagsWithDefaultValue:
                single_row_df[col] = tagsWithDefaultValue[col]

        df_input=single_row_df[feature_names]
        df_input=df_input.reindex(columns=feature_names)

        # print(df_input)
        single_row_dict = df_input.to_dict(orient='records')[0]
        df_input = model.named_steps['scaler'].transform(df_input)
        base_rf =model.named_steps['stacked_regressor'].estimators[0][1] 
        xgb_model = model.named_steps['stacked_regressor'].estimators[1][1]
        meta_model = model.named_steps['stacked_regressor'].final_estimator
        # Generate predictions from base models
        preds_base_rf = base_rf.predict(df_input)
        preds_base_xgb = xgb_model.predict(df_input)
        stacked_new_predictions = np.hstack([preds_base_rf.reshape(-1, 1), preds_base_xgb.reshape(-1, 1)])
        prediction= meta_model.predict(stacked_new_predictions)
        prediction_list = prediction.tolist()

        # Create the JSON response
        response = jsonify({
            "runningPv_tags": single_row_dict, "predictedPv": prediction_list, "missing_tags": missing_columns, "runningPv": runningpv_tag_dict,    # Dictionary of running PV tags
            "TS": ts_value  })
        # Return the response
        mins_pv_result = response
        return response
    
# @app.route('/api/start_5min_pv_scheduler', methods=['GET'])    
# def start_5min_pv_scheduler():
#     # with app.app_context():
#         # if not is_scheduler_running(scheduler):
#         #         start_scheduler()
#         #         return jsonify({"message": "Scheduler started successfully."}), 200
#         # else:
#         #     print('Scheduler is already running.')
#         #     return jsonify({"message": "Scheduler is already running."}), 200


@app.route('/api/running_pv_tag', methods=['GET'])
def running_pv_tag():
    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    # KUSTO_CLUSTER = "https://dfazuredataexplorer.westeurope.kusto.windows.net/"

    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"

    # KCSB = KustoConnectionStringBuilder.with_aad_device_authentication(KUSTO_CLUSTER)
    # KCSB.authority_id = AAD_TENANT_ID

    # KUSTO_CLIENT = KustoClient(KCSB)

    # In case you want to authenticate with AAD username and password
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)
    # active_tags = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2015_PVSM_MASS_EXIT_TEMP'
    # KUSTO_QUERY = f'{active_tags} set notruncation;Common2 | where SiteId == "LLPL" | where TS >= ago(15m) | where TS <= now()  | where Tag in ({result_tags_num})| summarize arg_max(TS, *) by Tag |project Tag, TS=datetime_add("minute", 330, TS), Value| evaluate pivot(Tag, max(Value))'
    # active_tag = 'TSPCAS3.Cascade3.PV_FROM_HMI'
    active_tag = 'TSPCAS3.Cascade3.PV_FROM_HMI'

    KUSTO_QUERY = f'''
    set notruncation;
    Common2 
    | where SiteId == "LLPL" 
    | where TS >= ago(15m) 
    | where TS <= now() 
    | where Tag == "{active_tag}"
    | summarize arg_max(TS, *) by Tag 
    | project Tag, cc, Value
    '''

    # print(KUSTO_QUERY)

    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])

    df=df.replace(r'^\s*$', np.nan, regex=True)
    df_sorted = df.sort_values(by='TS', ascending=True)
    # Ensure 'TS' column is in datetime format
    df_sorted['TS'] = pd.to_datetime(df_sorted['TS'])
    # Split the 'TS' column into 'Date' and 'Time' columns
    df_sorted['Date'] = df_sorted['TS'].dt.date
    df_sorted['Time'] = df_sorted['TS'].dt.strftime('%H:%M')

    # Rearrange columns for better readability
    df_15min = df_sorted[['Date', 'Time', 'Tag', 'Value']]

    # print(df_15min)
    # Convert DataFrame to list of dictionaries for JSON serialization
    df_15_min_intervals_dict = df_15min.fillna('').to_dict(orient='records')
    return jsonify({"predicted_pv":df_15_min_intervals_dict})

@app.route('/api/pv_by_tag_15mins', methods=['POST'])
def pv_by_tag_15mins():
    input_data = request.json
    # Access the values
    tag = input_data.get('tag')
    if tag == 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV':
         tag = 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI'
    # df_input=pd.DataFrame(input_data)
    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    # KUSTO_CLUSTER = "https://dfazuredataexplorer.westeurope.kusto.windows.net/"

    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"

    # In case you want to authenticate with AAD username and password
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)
    KUSTO_QUERY = f'''
    set notruncation;
    Common2 
    | where SiteId == "LLPL" 
    | where TS >= ago(15m) 
    | where TS <= now() 
    | where Tag == "{tag}"
    | project Tag, TS = datetime_add("minute", 330, TS), Value
    '''
    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
    # print(RESPONSE)
    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])
    # Check if the DataFrame is empty
    if df.empty:
        # If the DataFrame is empty, return the tag with a value of 0
        print('no data')
        if tag == 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI':
           tag = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'
        return jsonify({"runningPv":[]})
    else:    
        # print(df)
        df=df.replace(r'^\s*$', np.nan, regex=True)
        df_sorted = df.sort_values(by='TS', ascending=True)
        # Ensure 'TS' column is in datetime format
        df_sorted['TS'] = pd.to_datetime(df_sorted['TS'])

        batch_tags =['LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
                'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
                'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
                'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
                'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
                ]
        # Replace NaN values with an empty string
        df_sorted.replace(np.nan, '', inplace=True)
        # print(df_sorted, 'dc')
          # Check if tag is in batch_tags
        if tag in batch_tags:
            df_sorted['extracted_value'] = df_sorted['Value'].apply(
                lambda x: re.search(r'PV:(\d+)', x).group(1) if isinstance(x, str) and re.search(r'PV:(\d+)', x) else ''
                )
            df_sorted.rename(columns={'extracted_value': 'Value'}, inplace=True)
            
        # print(df_sorted)
        # Split the 'TS' column into 'Date' and 'Time' columns
        df_sorted['Date'] = df_sorted['TS'].dt.date
        df_sorted['Time'] = df_sorted['TS'].dt.strftime('%H:%M')


        # Rearrange columns for better readability
        df_15min = df_sorted[['Date', 'Time', 'Tag', 'Value']]
        df_15min['Tag'] = df_15min['Tag'].replace('TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI', 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV')

        # single_row_dict = single_row_df.to_dict(orient='records')[0]
        # Convert the 'TS' column to datetime if it exists
        if 'TS' in df_15min.columns:
            df_15min['TS'] = pd.to_datetime(df_15min['TS'])
        # print(df_15min)
        # Convert DataFrame to list of dictionaries for JSON serialization
        df_15_min_intervals_dict = df_15min.fillna('').to_dict(orient='records')
        return jsonify({"runningPv":df_15_min_intervals_dict})

@app.route('/api/pv_by_tag_timeline', methods=['POST'])
def pv_by_tag_timeline():
    input_data = request.json
    # Access the values
    tag = input_data.get('tag')

    if tag == 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV':
         tag = 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI'

    timeline = input_data.get('timeline')
    # Convert timeline to integer if possible
    try:
        timeline_int = int(timeline)
    except (ValueError, TypeError):
        timeline_int = None

    if timeline_int == 8:
        # Set the IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        # Get the current time in IST
        current_time = datetime.now(ist)
        # Calculate 8 hours ago in IST
        eight_hours_ago = current_time - timedelta(hours=8)
        # Set the time to 6 AM today in IST
        # Determine the start time based on the condition
        start_time = eight_hours_ago
        # The end time is the current time in IST
        end_time = current_time
        # Convert start and end times to UTC
        start_time_utc = start_time.astimezone(timezone.utc)
        end_time_utc = end_time.astimezone(timezone.utc)
        # Format the start and end times as strings
        start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
      
    elif isinstance(timeline_int, int) and timeline_int > 0 and timeline_int < 8:
        start_time_str = f"ago({timeline_int}h)"
        end_time_str = "now()"

    elif isinstance(timeline, str):  # Assuming timeline is in date format 'YYYY-MM-DD'
        try:
            timeline_date = datetime.strptime(timeline, '%Y-%m-%d')
            # Apply the IST timezone
            ist = pytz.timezone('Asia/Kolkata')
            start_time = timeline_date.replace(hour=6, minute=0, second=0, microsecond=0)
            start_time = ist.localize(start_time)

            # Convert the timeline_date to IST timezone
            timeline_date_ist = ist.localize(timeline_date)
            # Get today's date in IST
            today_date_ist = datetime.now(ist).date()

            # Check if timeline_date is today's date
            if timeline_date_ist.date() == today_date_ist:
                print("The timeline date is today's date.")
                # Calculate the end time
                end_time = datetime.now(ist)
            else:
                # Calculate the end time
                end_time = start_time + timedelta(days=1)
            
            # Convert to UTC
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            # Formatting the datetime to string for the query
            start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

        except ValueError:
            raise ValueError("Invalid date format for timeline. Please use 'YYYY-MM-DD'.")
    
    print("Start Time (UTC):", start_time_str)
    print("End Time (UTC):", end_time_str)
    # df_input=pd.DataFrame(input_data)
    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    # KUSTO_CLUSTER = "https://dfazuredataexplorer.westeurope.kusto.windows.net/"

    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"

    # In case you want to authenticate with AAD username and password
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)
    
    if isinstance(timeline_int, int) and timeline_int > 0 and timeline_int < 8:
        KUSTO_QUERY = f'''
            set notruncation;
            Common2 
            | where SiteId == "LLPL" 
            | where TS >= {start_time_str}
            | where TS <= {end_time_str}
            | where Tag == "{tag}"
            | project Tag, TS = datetime_add("minute", 330, TS), Value
            '''
    else:
        KUSTO_QUERY = f'''
            set notruncation;
            Common2 
            | where SiteId == "LLPL" 
            | where TS between (datetime({start_time_str}) .. datetime({end_time_str}))
            | where Tag == "{tag}"
            | project Tag, TS = datetime_add("minute", 330, TS), Value
            '''
    # print(KUSTO_QUERY)
    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
    # print(RESPONSE)
    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])
    # print(df)
    if df.empty:
        # If the DataFrame is empty, return the tag with a value of 0
        print('no data')
        if tag == 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI':
           tag = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'
        return jsonify({"runningPv":[]})
    else: 
        # print(df)
        df=df.replace(r'^\s*$', np.nan, regex=True)
        df_sorted = df.sort_values(by='TS', ascending=False)
        # Ensure 'TS' column is in datetime format
        df_sorted['TS'] = pd.to_datetime(df_sorted['TS'])
        batch_tags =['LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
                        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
                        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
                        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
                        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
                        ]
        # Replace NaN values with an empty string
        df_sorted.replace(np.nan, '', inplace=True)
        # print(df_sorted, 'dc')
          # Check if tag is in batch_tags
        if tag in batch_tags:
            df_sorted['extracted_value'] = df_sorted['Value'].apply(
                lambda x: re.search(r'PV:(\d+)', x).group(1) if isinstance(x, str) and re.search(r'PV:(\d+)', x) else ''
                )
            df_sorted = df_sorted[[ 'TS', 'Tag', 'extracted_value']]
            df_sorted.rename(columns={'extracted_value': 'Value'}, inplace=True)
        # Resample to 15-minute intervals and take the last record for each interval
        df_15min = df_sorted.resample('15T', on='TS').agg({'TS': 'last', 'Value': 'last', 'Tag': 'last'}).dropna().reset_index(drop=True)
        # Split the 'TS' column into 'Date' and 'Time' columns
        df_15min['Date'] = df_15min['TS'].dt.date
        df_15min['Time'] = df_15min['TS'].dt.strftime('%H:%M')

        # Rearrange columns for better readability
        df_15min['Tag'] = df_15min['Tag'].replace('TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI', 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV')

        # print(df_15min)
        # Convert DataFrame to list of dictionaries for JSON serialization
        df_15_min_intervals_dict = df_15min.fillna('').to_dict(orient='records')
        return jsonify({"runningPv":df_15_min_intervals_dict})

@app.route('/api/generate_alerts', methods=['GET'])
def generate_alerts():
     
    if not scheduler.running:
        scheduler.add_job(job, 'interval', minutes=5, id='my_job')
        scheduler.start()
        logger.info("Scheduler started successfully.")
    else:
        logger.info("Scheduler is already running.")

    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)
    active_tags = [
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
        ]

    # Tag_list = ['Tag_abc', 'Tag_def', 'Tag_dsd', 'tag_asds', 'Tag_asad']

    result_tags_num = ", ".join([f"Tag{i}" for i in range(1, len(active_tags) + 1)])

    # print(f'result_tags_num: {result_tags_num}')

    # Tag_list = ['Tag_abc', 'Tag_def', 'Tag_dsd', 'tag_asds', 'Tag_asad']

    result_tags = ""
    for i, tag in enumerate(active_tags, 1):
        result_tags += f"let Tag{i} = \"{tag}\"; "

    # print(f"result_tags :{result_tags}")
    KUSTO_QUERY = f'{result_tags} set notruncation;Common2 | where SiteId == "LLPL" | where TS >= ago(30m) | where TS <= now()  | where Tag in ({result_tags_num}) |project Tag, TS=datetime_add("minute", 330, TS), Value'

        
    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)

    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])
     # Initialize an empty list to store the sentences
    sentences = []
    displayData = [
                    {
                        'displayName': 'Noodler Chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',  'machineName': 'SIGMA-NOODLER', 'isControllable':'true'
                    },
                    {
                        'displayName': 'Noodler Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',  'machineName': 'SIGMA-NOODLER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Pre Plodder Vaccum', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Pre Plodder Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP', 'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Pre Plodder Chilled Water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Final Plodder Outlet Jacket Temperature (Turbo)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Final Plodder Flow Rate Outlet Flow (Litre/Hour)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Final Plodder Cone Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                    {
                        'displayName': 'Final plodder chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
                    },
                ]

    # Check if the DataFrame is empty
    if df.empty:
        # If the DataFrame is empty, return the tag with a value of 0
        print('no data')
        for data in displayData:
            if data['SensorName'] in active_tags:
                entry = {
                            'machineName': data['machineName'],
                            'parameterName': data['displayName'],
                            'reason': 'has no value',
                            'time': '30 min'
                        }
                sentences.append(entry)    
        return jsonify({"out_of_range_alerts":sentences})
    else:
        # print(df)
        df=df.replace(r'^\s*$', np.nan, regex=True)
        df_sorted = df.sort_values(by='TS', ascending=False)
        # print(df_sorted)
        tags_with_zero_or_empty_values = []
        # Step 1: Find unique tags present in the DataFrame
        tags_in_df = df['Tag'].unique()

        # Step 2: Find tags in active_tags that are not in tags_in_df
        missing_tags = [tag for tag in active_tags if tag not in tags_in_df]

        
        # print('missing_columns', missing_tags)
        # Append missing tags to the list
        for tag in missing_tags:
            tags_with_zero_or_empty_values.append(tag)

        column_values = pd.DataFrame(df_sorted)

        column_values.fillna(0, inplace=True)
        
        # Convert the entire DataFrame into a dictionary with columns as keys and lists of values as values
        all_values = {column: column_values[column].tolist() for column in column_values.columns}
        all_values_df = pd.DataFrame(all_values)

        batch_tags =['LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
                    ]
   
        
        pattern = r'PV:(\d+(\.\d+)?)'

        df['Value'] = df.apply(
            lambda row: re.search(pattern, row['Value']).group(1) if isinstance(row['Value'], str) and row['Tag'] in batch_tags and re.search(pattern, row['Value']) else row['Value'],
            axis=1
        )

        # Initialize a dictionary to store values by tag
        tag_values = {}

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            tag = row['Tag']
            value = row['Value']
            
            # Check if the tag already exists in the dictionary
            if tag not in tag_values:
                tag_values[tag] = []
            
            # Append the value to the list for this tag
            tag_values[tag].append(value)

        # Print the resulting dictionary
        # print("tag values", tag_values)
        for tag, values in tag_values.items():
            if all(pd.isna(value) or value == "" for value in values):
                tags_with_zero_or_empty_values.append(tag)

        # Output tags with all values zero or empty
        # print("Tags with all values zero or empty:", tags_with_zero_or_empty_values)
        # Convert the 'Value' column to float
        df['Value'] = df['Value'].astype(float)
        # Define recommended ranges for each tag
        recommended_ranges = {
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': (7.1789358712, 7.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': (29.79365892, 30.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': (-559.6211734, -556.25),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': (15.31891403, 16.25),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': (7.122084567, 7.7),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': (16.72576533, 17.4),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': (3995.517276, 4042.42),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': (51.48629739, 52),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': (7.384803243, 8.47),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': (28.14682946, 28.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': (10.8615525, 11.6),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': (17.9431487, 18.2),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': (102.3252551, 102.55),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': (458.214723, 458.6),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': (41.97259479, 43),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': (45.62208457, 46.2),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': (38.01472304, 38.4),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': (33.38261663, 33.8),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': (45.65419098, 46.2),
        }

        # Initialize an array to store tags with values out of the recommended range
        out_of_range_tags = []

        # Check each tag's values against the recommended range
        for tag, range_values in recommended_ranges.items():
            min_val, max_val = range_values
            if tag in tag_values:
                # Retrieve the values for the current tag
                values = tag_values[tag]
                
                # Convert string values to floats and check if they are all out of the recommended range
                if all((float(val) < min_val or float(val) > max_val) for val in values):
                    out_of_range_tags.append(tag)

        # Output the tags with out-of-range values
        # print("Tags with all out of recommended range tags:", out_of_range_tags)

        # Iterate through out_of_range_tags and displayData to create sentences
        for tag in out_of_range_tags:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'is out of recommended range',
                                'time': '30 min'
                            }
                    sentences.append(entry)

                    # sentence = f"machineName : {data['machineName']} , parameterName : {data['displayName']} , reason: is out of recommended range, time: 30 min"
                    # sentences.append(sentence)
        
        # Iterate through out_of_range_tags and displayData to create sentences
        for tag in tags_with_zero_or_empty_values:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'has no value',
                                'time': '30 min'
                            }
                    # sentence = f"machineName : {data['machineName']}, parameterName : {data['displayName']}, reason: is empty, time: 30 min"
                    sentences.append(entry)
        # print(df_15_min)
        return jsonify({"out_of_range_alerts":sentences})

@app.route('/api/generate_insights', methods=['POST'])
def generate_insights():

    input_data = request.json
    # Access the values
    active_tags = input_data.get('tag')
    manual_tag = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'
    replacement_tag  = 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI'
    if manual_tag in active_tags:
        active_tags = [replacement_tag if tag == manual_tag else tag for tag in active_tags]
    # print('tags', active_tags)
    timeline = input_data.get('timeline')
    # Convert timeline to integer if possible
    try:
        timeline_int = int(timeline)
    except (ValueError, TypeError):
        timeline_int = None
    timeline_data = ""
    if timeline_int == 8:
        current_time = datetime.now()
        # Calculate 8 hours ago
        start_time = current_time - timedelta(hours=8)   
        end_time = current_time
        start_time_utc = start_time.astimezone(timezone.utc)
        end_time_utc = end_time.astimezone(timezone.utc)
        start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        timeline_data = "8 Hour"
    elif isinstance(timeline_int, int) and timeline_int > 0 and timeline_int < 8:
        start_time_str = f"ago({timeline_int}h)"
        end_time_str = "now()"
        timeline_data = f"{timeline_int} Hour" 
    elif isinstance(timeline_int, int) and timeline_int == 15:
        start_time_str = f"ago({timeline_int}m)"
        end_time_str = "now()"    
        timeline_data = "15 Min"
    elif isinstance(timeline, str):  # Assuming timeline is in date format 'YYYY-MM-DD'
        try:
            timeline_date = datetime.strptime(timeline, '%Y-%m-%d')
            start_time = timeline_date.replace(hour=6, minute=0, second=0, microsecond=0)
            start_time_utc = start_time.astimezone(timezone.utc)
            ist = pytz.timezone('Asia/Kolkata')
             # Convert the timeline_date to IST timezone
            timeline_date_ist = ist.localize(timeline_date)
            # Get today's date in IST
            today_date_ist = datetime.now(ist).date()

            # Check if timeline_date is today's date
            if timeline_date_ist.date() == today_date_ist:
                print("The timeline date is today's date.")
                # Calculate the end time
                end_time = datetime.now(ist)
            else:
                # Calculate the end time
                end_time = start_time + timedelta(days=1)

            end_time_utc = end_time.astimezone(timezone.utc)
            start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            formatted_start_time = start_time.strftime('%d %b %Y, %I:%M %p')
            formatted_end_time = end_time.strftime('%d %b %Y, %I:%M %p')
            timeline_data = f'''{formatted_start_time} to {formatted_end_time}'''
        except ValueError:
            raise ValueError("Invalid date format for timeline. Please use 'YYYY-MM-DD'.")
    
    print("Start Time (UTC):", start_time_str)
    print("End Time (UTC):", end_time_str)

    AAD_TENANT_ID = "f66fae02-5d36-495b-bfe0-78a6ff9f8e6e"
    KUSTO_CLUSTER= "https://dfazuredataexplorer.westeurope.kusto.windows.net/"
    KUSTO_DATABASE = "dfdataviewer"
    username = "ashish.mishra2@unilever.com"
    password = "SitaRamRadha@1"
    authority_id = AAD_TENANT_ID
    KCSB = KustoConnectionStringBuilder.with_aad_user_password_authentication(KUSTO_CLUSTER, username, password, authority_id)
    KUSTO_CLIENT = KustoClient(KCSB)
   
    result_tags_num = ", ".join([f"Tag{i}" for i in range(1, len(active_tags) + 1)])
    
    result_tags = ""
    for i, tag in enumerate(active_tags, 1):
        result_tags += f"let Tag{i} = \"{tag}\"; "

    # KUSTO_QUERY = f'{result_tags} set notruncation;Common2 | where SiteId == "LLPL" | where TS >= ago(30m) 
    # | where TS <= now()  | where Tag in ({result_tags_num}) |project Tag, TS=datetime_add("minute", 330, TS), Value'
    
    if isinstance(timeline_int, int) and timeline_int > 0 and timeline_int < 8 or timeline_int == 15:
        KUSTO_QUERY = f'''{result_tags}
            set notruncation;
            Common2 
            | where SiteId == "LLPL" 
            | where TS >= {start_time_str}
            | where TS <= {end_time_str}
            | where Tag in ({result_tags_num})
            | project Tag, TS = datetime_add("minute", 330, TS), Value
            '''
    else:
        KUSTO_QUERY = f'''{result_tags}
            set notruncation;
            Common2 
            | where SiteId == "LLPL" 
            | where TS between (datetime({start_time_str}) .. datetime({end_time_str}))
            | where Tag in ({result_tags_num})
            | project Tag, TS = datetime_add("minute", 330, TS), Value
            '''
    # print(KUSTO_QUERY)    
    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)

    df = pd.DataFrame()
    df = dataframe_from_result_table(RESPONSE.primary_results[0])
    manual_tag = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'
    replacement_tag  = 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI'
    if replacement_tag in active_tags:
        active_tags = [manual_tag if tag == replacement_tag else tag for tag in active_tags]
    sentences = []
    displayData = [
            {
                'displayName': 'Noodler Chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',  'machineName': 'SIGMA-NOODLER', 'isControllable':'true'
            },
            {
                'displayName': 'Noodler Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',  'machineName': 'SIGMA-NOODLER', 'isControllable': 'true'
            },
            {
                'displayName': 'Pre Plodder Vaccum', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Pre Plodder Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP', 'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Pre Plodder Chilled Water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Final Plodder Outlet Jacket Temperature (Turbo)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Final Plodder Flow Rate Outlet Flow (Litre/Hour)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Final Plodder Cone Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'Final plodder chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',  'machineName': 'FINAL-PLODDER', 'isControllable': 'true'
            },
            {
                'displayName': 'PSM Noodle Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP',  'machineName': 'SIGMA-MIXER', 'isControllable': 'false'
            },
            {
                'displayName': 'Sigma PSM Hot Water PV', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV',  'machineName': 'SIGMA-MIXER', 'isControllable': 'false'
            },
            {
                'displayName': 'Moisture In Noodles (Moisture Meter)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV', 'machineName': 'SIGMA-MIXER', 'isControllable': 'false'
            }, 
            {
                'displayName': 'Sigma RM Starch', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH', 'isControllable': 'false', 'machineName': 'SIGMA-MIXER'
            }, 
            {
                'displayName': 'Sigma RM Noodle', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle', 'isControllable': 'false', 'machineName': 'SIGMA-MIXER'
            },
            {
                'displayName': 'Mixer Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP', 'isControllable': 'false', 'machineName': 'SIGMA-MIXER'
            },
            {
                'displayName': 'Noodler Soap Mass Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP', 'machineName': 'SIGMA-NOODLER', 'isControllable': 'false'
            },
            {
                'displayName': 'TRM Soap Mass Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP', 'machineName': 'TRIPPLE-ROLLER-MILL', 'isControllable': 'false'
            },
            {
                'displayName': 'Final Plodder Motor Current', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005', 'machineName': 'FINAL-PLODDER', 'isControllable': 'false'
            },
            {
                'displayName': 'Final Plodder Bar Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP', 'machineName': 'FINAL-PLODDER', 'isControllable': 'false'
            },
        ]

    # Check if the DataFrame is empty
    if df.empty:
        print('no data')
        # If the DataFrame is empty, return the tag with a value of 0
        # for data in displayData:
        #         if data['SensorName'] in active_tags:
        #             entry = {
        #                         'machineName': data['machineName'],
        #                         'parameterName': data['displayName'],
        #                         'reason': 'has no value',
        #                         'time': timeline_data
        #                     }
        #             sentences.append(entry)    
        return jsonify({"insights":sentences, "dataAvaiable" : "false"})
    else:
        df=df.replace(r'^\s*$', np.nan, regex=True)
        df['Tag'] = df['Tag'].replace('TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI', 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV')
        df_sorted = df.sort_values(by='TS', ascending=False)
        # print(df)
        # if tag == 'TSPCAS3.Cascade3.NOODLER_MIV_FROM_HMI':
        #     tag = 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV'
        column_values = pd.DataFrame(df_sorted)

        column_values.fillna(0, inplace=True)
        # print('cv', column_values)
        # Convert the entire DataFrame into a dictionary with columns as keys and lists of values as values
        all_values = {column: column_values[column].tolist() for column in column_values.columns}
        all_values_df = pd.DataFrame(all_values)
        # Convert the dictionary into a DataFrame
        # print(all_values_df)
        batch_tags =['LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_batch_time',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Liquid_Minor',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_Perfume',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH',
                    'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle',
                    ]
   
       
        pattern = r'PV:(\d+(\.\d+)?)'
        # print(df['Value'])
        
        df['Value'] = df.apply(
            lambda row: re.search(pattern, row['Value']).group(1) if isinstance(row['Value'], str) and row['Tag'] in batch_tags and re.search(pattern, row['Value']) else row['Value'],
            axis=1
        )
        # Initialize a dictionary to store values by tag
        tag_values = {}

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            tag = row['Tag']
            value = row['Value']
            
            # Check if the tag already exists in the dictionary
            if tag not in tag_values:
                tag_values[tag] = []
            
            # Append the value to the list for this tag
            tag_values[tag].append(value)

        # Print the resulting dictionary
        # print('tag_values', tag_values)
        tags_with_zero_or_empty_values = []
        for tag, values in tag_values.items():
            if all(pd.isna(value) or value == "" for value in values):
                tags_with_zero_or_empty_values.append(tag)

        # Convert the 'Value' column to float
        df['Value'] = df['Value'].astype(float)
        # Define recommended ranges for each tag
        recommended_ranges = {
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': (7.1789358712, 7.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': (29.79365892, 30.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': (-559.6211734, -556.25),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': (15.31891403, 16.25),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': (7.122084567, 7.7),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': (16.72576533, 17.4),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': (3995.517276, 4042.42),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': (51.48629739, 52),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': (7.384803243, 8.47),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': (28.14682946, 28.5),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': (10.8615525, 11.6),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': (17.9431487, 18.2),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': (102.3252551, 102.55),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': (458.214723, 458.6),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': (41.97259479, 43),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': (45.62208457, 46.2),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': (38.01472304, 38.4),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': (33.38261663, 33.8),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': (45.65419098, 46.2),
        }

        # Initialize an array to store tags with values out of the recommended range
        out_of_recommended_range_tags = []

        # Check each tag's values against the recommended range
        for tag, range_values in recommended_ranges.items():
            min_val, max_val = range_values
            
            if tag in tag_values:
                # Retrieve the values for the current tag
                values = tag_values[tag]
                
                # Convert string values to floats and check if they are all out of the recommended range
                if all((float(val) < min_val or float(val) > max_val) for val in values):
                    out_of_recommended_range_tags.append(tag)

        # Output the tags with out-of-range values
        # print("Tags with all out of recommended range tags:", out_of_recommended_range_tags)
        
        # Define business ranges for each tag
        business_ranges = {
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': (7, 10),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': (20, 30),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': (-600, -450),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': (15, 30),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': (6, 20),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': (10, 20),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': (2000, 3000),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': (40, 60),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': (7, 10),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': (20, 30),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': (10, 20),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': (17, 20),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': (40, 60),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': (450, 500),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': (20, 40),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': (40, 45),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': (40, 45),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': (30, 35),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': (40, 50),
            }
        
        # Initialize an array to store tags with values out of the business range
        out_of_business_range_tags = []

        # Check each tag's values against the recommended range
        for tag, range_values in business_ranges.items():
            min_val, max_val = range_values
            
            if tag in tag_values:
                # Retrieve the values for the current tag
                values = tag_values[tag]
            
                # Check if all values are out of the recommended range
                if all((float(val) < min_val or float(val) > max_val) for val in values):
                    out_of_business_range_tags.append(tag)

        # Output the tags with out-of-range values
        # print("Tags with all out of business range tags:", out_of_business_range_tags)

        # Define outliner ranges for each tag
        outliner_ranges = {
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': (6.5, 16),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': (10, 40),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': (-700, -400),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': (10, 40),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': (6, 30),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': (10, 35),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': (2000, 4500),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': (30, 70),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': (6.5, 20),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': (20, 40),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': (5, 30),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': (15, 25),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': (30, 105),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': (400, 600),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': (10, 60),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': (30, 50),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': (30, 50),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': (25, 40),
            'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': (45, 55),
            }
        
        # Initialize an array to store tags with values out of the outliner range
        out_of_outliner_range_tags = []

        # Check each tag's values against the outliner range
        for tag, range_values in outliner_ranges.items():
            min_val, max_val = range_values
            
            if tag in tag_values:
                # Retrieve the values for the current tag
                values = tag_values[tag]
            
                # Check if all values are out of the recommended range
                if all((float(val) < min_val or float(val) > max_val) for val in values):
                    out_of_outliner_range_tags.append(tag)

        # Output the tags with out-of-range values
        print("Tags with all out of outliner range tags:", out_of_outliner_range_tags)

        # Iterate through out_of_range_tags and displayData to create sentences
        for tag in out_of_recommended_range_tags:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'is out of recommended range',
                                'time': timeline_data
                            }
                    sentences.append(entry)
        # Iterate through out_of_range_tags and displayData to create sentences
        for tag in out_of_business_range_tags:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'is out of business range',
                                'time': timeline_data
                            }
                    sentences.append(entry)
        # Iterate through out_of_range_tags and displayData to create sentences
        for tag in out_of_outliner_range_tags:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'is out of outliner range',
                                'time': timeline_data
                            }
                    sentences.append(entry)
         # Iterate through out_of_range_tags and displayData to create sentences
        for tag in tags_with_zero_or_empty_values:
            for data in displayData:
                if data['SensorName'] == tag:
                    entry = {
                                'machineName': data['machineName'],
                                'parameterName': data['displayName'],
                                'reason': 'has no value',
                                'time': timeline_data
                            }
                    sentences.append(entry)
        return jsonify({"insights":sentences, "dataAvaiable" : "true"})

container_name = "unilever"
blob_name = "Llpl-Predicted-Pv/history_of_pv_records.json"
# blob_service_client = get_blob_access()

def get_blob_access() -> BlobServiceClient:
    # print('get_blob_access')
    global blob_service_client
    # Define the SAS token and the blob URL
    # sas_token = "sp=rwdl&sv=2020-12-06&sdd=1&spr=https&sig=vIBrTmDOCoUGEOCku98tmLjobfqGSxiJO%2bRB48wyIY8%3d&si=2024-Q4&sr=d"
    sas_token = "sp=rwdl&sv=2020-12-06&sdd=1&spr=https&sig=LTNRxenI3uxr63nB6cwJbfqnLJwnmyxYeMtcyTx1Lwg%3d&si=2025-Q3&sr=d"
    blob_url = "https://dbstorageda16d88047adls.blob.core.windows.net/unilever/Llpl-Predicted-Pv?" + sas_token

    # sas_token = "https://dbstorageda16d88047adls.blob.core.windows.net/unilever/Llpl-Predicted-Pv?sp=rwdl&sv=2020-12-06&sdd=1&spr=https&sig=lIwP62%2bOVvAjYR%2fM%2f3AWc9qX3m9G%2fRnb3G4RDlhrGyg%3d&si=2024-Q3&sr=d"
    # account_url = "https://dbstorageda16d88047adls.blob.core.windows.net"
    # Create a BlobServiceClient using the blob URL with the SAS token
    # Check if blob_service_client is None
    if blob_service_client is None:
        try:
            # Construct the BlobServiceClient with the account URL and SAS token
            blob_service_client = BlobServiceClient(account_url=blob_url)
            return blob_service_client
        except Exception as e:
            print(f"Error while creating BlobServiceClient: {e}")
            return None
    # print('blob_service_client', blob_service_client)
    return blob_service_client
   
def fetch_records():
    print('fetch method calling')
    with app.app_context():
    # Call the Flask route function directly
        mins_pv_result = predicted_pv()
        # Convert response object to JSON
        if mins_pv_result is not None:
            mins_pv_result_json = mins_pv_result.get_json()

            # The original IST time (assuming it's in IST)
            ist_time_str = mins_pv_result_json.get('TS')
            input_runningPv = mins_pv_result_json.get('runningPv')
            input_predictedPv = mins_pv_result_json.get('predictedPv')
            # Convert the ISO 8601 IST time string to a datetime object
            ist_time = datetime.fromisoformat(ist_time_str)
            # Subtract 330 minutes (5 hours and 30 minutes) to convert IST to UTC
            utc_time = ist_time - timedelta(minutes=330)
            # Convert the UTC datetime object to ISO format with microseconds, then add 'Z'
            utc_time_str = utc_time.isoformat() + "Z"
            print(utc_time_str)
            # Define your local time in IST (without UTC offset initially)
            records = {
                "TS": utc_time_str, 
                "runningPv": input_runningPv, 
                "predictedPv": input_predictedPv
            }

            # Check if 'predictedPv' is a list and contains exactly one item
            if isinstance(records.get('predictedPv'), list) and len(records['predictedPv']) == 1:
                records['predictedPv'] = records['predictedPv'][0]
            
            # update_blob(records)
            return records

def update_blob(records):
    print('update method calling', records)
    
    # Create or update the blob with the records
    # Download the blob
    blob_service_client = get_blob_access()
    # container_client = blob_service_client.get_container_client(container_name)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    # Check if the blob already exists and read its content
    try:
        existing_blob = blob_client.download_blob().readall()
        existing_data = json.loads(existing_blob)
    except Exception as e:
        # Blob does not exist or other error occurred; initialize an empty list
        existing_data = []
    # print('existing_data', existing_data)
    # Append new records
    # Check for duplicates before appending
    if records not in existing_data:
        existing_data.append(records)
    
    # Upload updated data to Blob Storage
    blob_client.upload_blob(json.dumps(existing_data), overwrite=True)
    # print(f"Updated {blob_name} with new records.")

def job():
    print('job method calling')
    # Fetch records based on your logic
    records = fetch_records()
    if records is None:
        return
    # Update the blob with the new records
    update_blob(records)

    # Global Scheduler Instance
# scheduler = BackgroundScheduler()

def is_scheduler_running(scheduler):
    # Check if scheduler has any active jobs or is running
    return len(scheduler.get_jobs()) > 0 or scheduler.running

def start_scheduler():
    # Scheduler Setup
    scheduler.add_job(job, 'interval', minutes=5)
    scheduler.start()
    print('Scheduler started. Waiting for jobs to run...')
    
    # Keep the script running to allow the scheduler to execute jobs
    try:
        while True:
            time.sleep(1)  # Sleep for a short period to avoid high CPU usage
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()  # Shut down the scheduler gracefully

def get_data_from_blob():
    try:
        # Download the blob content
        blob_service_client = get_blob_access()
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        stream_downloader = blob_client.download_blob()
        data = stream_downloader.readall()
        # Convert the JSON data to a Pandas DataFrame
        df = pd.read_json(io.BytesIO(data))
        return df
    
    except Exception as e:
        print(f"Error fetching data from Blob Storage: {str(e)}")
        return pd.DataFrame()

@app.route('/api/get_pv_values', methods=['POST'])
def get_pv_values():
    # Retrieve the timestamp from query parameters
    input_data = request.json
    timeline = input_data.get('timeline')
    # print(timeline)
    if not timeline:
        return jsonify({"error": "timeline parameter is required"}), 400
    
    try:
        try:
            timeline_int = int(timeline)
        except (ValueError, TypeError):
            timeline_int = None
        if timeline_int == 8:
            current_time = datetime.now()
            # Calculate 8 hours ago
            start_time = current_time - timedelta(hours=8)   
            end_time = current_time
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # timeline_data = "8 Hour"
        elif isinstance(timeline_int, int) and timeline_int > 0 and timeline_int < 8:
            # start_time_str = f"ago({timeline_int}h)"
            current_time = datetime.now()
            start_time = current_time - timedelta(hours=timeline_int)   
            end_time = current_time
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # end_time_str = "now()"
            # timeline_data = f"{timeline_int} Hour" 
        elif isinstance(timeline_int, int) and timeline_int == 15:
            # start_time_str = f"ago({timeline_int}m)"
            current_time = datetime.now()
            start_time = current_time - timedelta(minutes=timeline_int)   
            end_time = current_time
            start_time_utc = start_time.astimezone(timezone.utc)
            end_time_utc = end_time.astimezone(timezone.utc)
            start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            # end_time_str = "now()"    
            # timeline_data = "15 Min"
        elif isinstance(timeline, str):  # Assuming timeline is in date format 'YYYY-MM-DD'
            try:
                ist = pytz.timezone('Asia/Kolkata')
                timeline_date = datetime.strptime(timeline, '%Y-%m-%d')
                start_time = timeline_date.replace(hour=6, minute=0, second=0, microsecond=0)
                start_time_ist = ist.localize(start_time)
                start_time_utc = start_time_ist.astimezone(pytz.UTC)
                # Convert the timeline_date to IST timezone
                timeline_date_ist = ist.localize(timeline_date)
                # Get today's date in IST
                today_date_ist = datetime.now(ist).date()

                # Check if timeline_date is today's date
                if timeline_date_ist.date() == today_date_ist:
                    print("The timeline date is today's date.")
                    # Calculate the end time
                    end_time = datetime.now(ist)
                else:
                    # Calculate the end time
                    end_time = start_time_utc + timedelta(days=1)

                end_time_utc = end_time.astimezone(timezone.utc)
                start_time_str = start_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                end_time_str = end_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                # formatted_start_time = start_time.strftime('%d %b %Y, %I:%M %p')
                # formatted_end_time = end_time.strftime('%d %b %Y, %I:%M %p')
                # timeline_data = f'''{formatted_start_time} to {formatted_end_time}'''
            except ValueError:
                raise ValueError("Invalid date format for timeline. Please use 'YYYY-MM-DD'.")
        
        print("Start Time (UTC):", start_time_str)
        print("End Time (UTC):", end_time_str)
        
    except ValueError:
        return jsonify({"error": "Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS format."}), 400
    
    # Read data from Azure Blob Storage
    df = get_data_from_blob()
    # print('df', df)
    if df.empty:
        print(jsonify({"error": "No data found or error in fetching data from storage."}), 200)
        return jsonify({'pv_values': []}), 200
    # Convert the 'TS' column to datetime
    # Define start and end times for filtering
    start_time = pd.to_datetime(start_time_str)  # Your start time
    end_time = pd.to_datetime(end_time_str)    # Your end time
    # Remove the trailing 'Z' character, if present
    df['TS'] = df['TS'].str.replace('Z$', '', regex=True)
    df['TS'] = pd.to_datetime(df['TS'])
    # Filter the DataFrame based on the time range
    filtered_df_data = df[(df['TS'] >= start_time) & (df['TS'] <= end_time)]
    # print('dfd', filtered_df_data)
    # print('le',len(filtered_df_data))
    if filtered_df_data.empty:
        return jsonify({"pv_values": []}), 200  
    else:
        # Resample every 15 minutes and take the last record
        # Convert the 'TS' column to datetime format and set timezone to UTC
        filtered_df_data['TS'] = pd.to_datetime(filtered_df_data['TS'], utc=True)

        # Convert from UTC to IST
        filtered_df_data['TS'] = filtered_df_data['TS'].dt.tz_convert('Asia/Kolkata')

        # Set 'TS' as the index and resample the data for 15-minute intervals
        filtered_df_data.set_index('TS', inplace=True)
        # if isinstance(timeline_int, int) and timeline_int == 15:
        #     filtered_df = filtered_df_data.resample('15T').last().reset_index()
        # else:
        # Remove duplicates from the DataFrame
        # print('filtered_df_data', filtered_df_data)
        filtered_df_data = filtered_df_data[~filtered_df_data.index.duplicated(keep='last')]
        # print('post filtered_df_data', filtered_df_data)
        filtered_df = filtered_df_data.resample('15T').last().reset_index()
        # Create tooltip columns based on whether the resampled values were NaN
        filtered_df['runningPv_tooltip'] = np.where(filtered_df['runningPv'].isna(), 'no data', '')
        filtered_df['predictedPv_tooltip'] = np.where(filtered_df['predictedPv'].isna(), 'no data', '')

        # Fill NaN values in runningPv and predictedPv columns with 0
        filtered_df['runningPv'] = filtered_df['runningPv'].fillna(0)
        filtered_df['predictedPv'] = filtered_df['predictedPv'].fillna(0)

        # Extract 'Date' and 'Time' from the 'TS' column
        filtered_df['Date'] = filtered_df['TS'].dt.date
        filtered_df['Time'] = filtered_df['TS'].dt.strftime('%H:%M')

        # Convert the filtered DataFrame to JSON
        result = filtered_df.to_dict(orient='records')
        return jsonify({"pv_values": result}), 200

if __name__ == "__main__":
    # Start the scheduler automatically when the app starts
    scheduler.add_job(job, 'interval', minutes=5, id='my_job')
    scheduler.start()
    logger.info("Scheduler started automatically on app initialization.")
    logger.info("Application started.")
    app.run(debug=True)
