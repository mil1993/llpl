import pandas as pd

def format_datetime(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value

feature_imp_list = [
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP'
]


simultion_data_list = [
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP',
'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP'
]

tagsWithDefaultValue = [
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP': '7.34'
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP': 30.15
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC': -557.94
    },
    {
       'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP': 15.78
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP': 7.41
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP': 17.06
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE': 4018.86
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP': 51.74
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP': 7.93
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP': 28.32
    },
    {
       'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV': 11.23
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV': 18.07
    }, 
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH': 102.44
    }, 
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle': 458.41
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP': 42.49
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP': 45.91
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP': 38.21
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005': 33.59
    },
    {
        'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP': 45.93
    }
]

# predictionData = [
#     {
#         displayName: 'Noodler Chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_IN_TEMP', InputValue: '7.34', unit: 'deg. C', lsl: 6.5, usl: 16, rlr: 7.1789358712, rur: 7.5, blr:7, bur:10, machineName: 'SIGMA-NOODLER-HOPPER', 'isControllable':'true'
#     },
#     {
#         displayName: 'Noodler Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_NOODLER_OUT_TEMP', InputValue: 30.15, unit: 'deg. C', lsl: 10, usl: 40, rlr: 29.79365892, rur: 30.5,  blr: 20, bur: 30, machineName: 'SIGMA-NOODLER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Pre Plodder Vaccum', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_VAC_CHAMBER_PVPLDR_VAC_CHAMBER_VAC', InputValue: -557.94, unit: 'mm HG', lsl: -700, usl: -400, rlr: -559.6211734, rur: -556.25, blr: -600, bur: -450, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Pre Plodder Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_OUT_TEMP', InputValue: 15.78, unit: 'deg. C', lsl: 10, usl: 40, rlr: 15.31891403, rur: 16.25, blr: 15, bur: 30, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Pre Plodder Chilled Water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_PRE_IN_TEMP', InputValue: 7.41, unit: 'deg. C', lsl: 6, usl: 30, rlr: 7.122084567, rur: 7.7, blr: 6, bur: 20, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Final Plodder Outlet Jacket Temperature (Turbo)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_OUT_TEMP', InputValue: 17.06, unit: 'deg. C', lsl: 10, usl: 35, rlr: 16.72576533, rur: 17.4, blr: 10, bur: 20, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Final Plodder Flow Rate Outlet Flow (Litre/Hour)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_FPLDR_BAR_FLOW_RATEFPLDR_BAR_FLOW_RATE', InputValue: 4018.86, unit: 'litre/hour', lsl: 2000, usl: 4500, rlr: 3995.517276, rur: 4042.2, blr: 2000, bur: 3000, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Final Plodder Cone Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2025_PVFPLDR_MOUTH_TEMP', InputValue: 51.74, unit: 'deg. C', lsl: 30, usl: 70, rlr: 51.48629739, rur: 52, blr: 40, bur: 60, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'Final plodder chilled water In temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_FINAL_IN_TEMP', InputValue: 7.93, unit: 'deg. C', lsl: 6.5, usl: 20, rlr: 7.384803243, rur: 8.47, blr: 7, bur: 10, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': true
#     },
#     {
#         displayName: 'PSM Noodle Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_TT_2011_PVPSM_NDLR_TEMP', InputValue: 28.32, unit: 'deg. C', lsl: 20, usl: 40, rlr: 28.14682946, rur: 28.5, blr: 20, bur: 30, machineName: 'SIGMA-MIXER', 'isControllable': false
#     },
#     {
#         displayName: 'Sigma PSM Hot Water PV', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_PSM_HOT_WATER_PV', InputValue: 11.23, unit: 'Kg', lsl: 5, usl: 30, rlr: 10.8615525, rur: 11.6, blr: 10, bur: 20, machineName: 'SIGMA-MIXER', 'isControllable': false
#     },
#     {
#         displayName: 'Moisture In Noodles (Moisture Meter)', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_TSP_CAS2_PWP_LLPL_800500066581_NDLS_MOIST_PVPSM_NDLS_MOIST_PV', InputValue: 18.07, unit: '%', lsl: 15, usl: 25, rlr: 17.9431487, rur: 18.2, blr: 17, bur: 20, machineName: 'SIGMA-MIXER', 'isControllable': false
#     }, 
#     {
#         displayName: 'Sigma RM Starch', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_PSM_batch_TSP_CAS2_PWP_LLPL_800500066581_RM_STARCH', InputValue: 102.44, unit: 'Kg', lsl: 30, usl: 105, rlr: 102.3252551, rur: 102.55, 'isControllable': false, blr: 40, bur: 60, machineName: 'SIGMA-MIXER'
#     }, 
#     {
#         displayName: 'Sigma RM Noodle', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigmamixer_batch_800500066581_Noodle', InputValue: 458.41, unit: 'Kg', lsl: 400, usl: 600, rlr: 458.214723, rur: 458.6, 'isControllable': false, blr: 450, bur: 500, machineName: 'SIGMA-MIXER'
#     },
#     {
#         displayName: 'Mixer Out Turbo', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade_2_TURBO_MIXER_OUT_TEMP', InputValue: 42.49, unit: 'deg. C', lsl: 10, usl: 60, rlr: 41.97259479, rur: 43, 'isControllable': false, blr: 20, bur: 40, machineName: 'SIGMA-MIXER'
#     },
#     {
#         displayName: 'Noodler Soap Mass Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2016_PVSM_NDLR_MASS_EXIT_TEMP', InputValue: 45.91, unit: 'deg. C', lsl: 30, usl: 50, rlr: 45.62208457, rur: 46.2, blr: 40, bur: 45, machineName: 'SIGMA-NOODLER-HOPPER', 'isControllable': false
#     },
#     {
#         displayName: 'TRM Soap Mass Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2019_PVTRM_MASS_EXIT_TEMP', InputValue: 38.21, unit: 'deg. C', lsl: 30, usl: 50, rlr: 38.01472304, rur: 38.4, blr: 40, bur: 45, machineName: 'TRIPPLE-ROLLER-MILL', 'isControllable': false
#     },
#     {
#         displayName: 'Final Plodder Motor Current', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.TSP_CAS2_PWP_LLPL_800500066581_MX02_Final_Plodder_PV_I_2005', InputValue: 33.59, unit: 'amp', lsl: 25, usl: 40, rlr: 33.38261663, rur: 33.8, blr: 30, bur: 35, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': false
#     },
#     {
#         displayName: 'Final Plodder Bar Temperature', 'SensorName': 'LOGIX_cas2_pwp_llpl.cas2_pwp_llpl.Cascade 2_Sigma_batch_TSP_CAS2_PWP_LLPL_800500066581_TT_2026_PVFPLDR_SOAP_TEMP', InputValue: 45.93, unit: 'deg. C', lsl: 45, usl: 55, rlr: 45.65419098, rur: 46.2, blr: 40, bur: 50, machineName: 'FINAL-PLODDER-HOPPER', 'isControllable': false
#     },

# ]

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




