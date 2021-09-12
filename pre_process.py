from collections import deque

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from helper import *
pd.options.mode.chained_assignment = None



def main():
    health_facility = pd.read_csv('health_facility_assessment.csv')

    # clean_column_names holds sanitized variable names
    clean_column_names = deque([])
    df = health_facility

    # drop the first 2 variables
    df = df.drop(health_facility.iloc[:, 0:2], axis=1)

    for name in list(df.columns):
        string_ele = name.split(".")
        if len(string_ele) > 2:
            new_name = '_'.join(string_ele[-2:])
            clean_column_names.append(new_name)
        else:
            clean_column_names.append(string_ele[-1])

    # rename old variables to new variable names
    df.columns = clean_column_names

    # Fill empty spaces/character with NaN
    df = df.apply(lambda x: x.str.strip() if isinstance(
        x, str) else x).replace('---', np.nan)


    # Find Null variables
    empty_columns = [column for column in df.columns if df[column].isnull().all()]
    empty_columns.append('health_centre_information_managing_authority_other')

    # Drop Null variables. Dropped 20 empty variables
    df = df.drop(empty_columns, axis=1)

    # Convert columns 11 - 31 to numerical variables
    df = column_to_numeric(df, 11, 31)
    # Convert columns 33 - 49 to numerical variables
    df = column_to_numeric(df, 33, 49)
    # Convert columns 152 - 185 to float variables
    df = column_to_float(df, 152, 185)

    # Cast columns to categorical variable
    df[['health_centre_information_managing_authority',
        'health_centre_information_setting',
        'health_centre_information_outpatient_only']].apply(lambda x: x.astype('category'))

    # Split GPS location into Latituded, Longitude to its column
    gps_df = df['facility_gps'].str.split(',', expand=True)\
                            .rename(columns={0: 'facility_latitude', 1: 'facility_longitude'})

    # Combine the two datasets
    df = pd.concat([df, gps_df], axis=1)

    # Drop rows
    df = df.dropna(subset=['health_centre_information_facility_name'], axis=0)

    # FILL numeric columns with empty values USING Multivariate imputation by chained equations (MICE)
    # ----> df.columns[df.isna().any()].tolist() # List all the columns with empty rows
    missing_numeric_columns = ['head_count_month_1',
                            'head_count_month_2',
                            'head_count_month_3',
                            'head_count_month_4',
                            'head_count_month_1',
                            'head_count_month_2',
                            'head_count_month_3',
                            'head_count_month_4',
                            'general_outpatient_month_1',
                            'general_outpatient_month_2',
                            'general_outpatient_month_3',
                            'general_outpatient_month_4',
                            'deliveries_month_1',
                            'deliveries_month_2',
                            'deliveries_month_3',
                            'deliveries_month_4',
                            'pent_vaccines_month_1',
                            'pent_vaccines_month_2',
                            'pent_vaccines_month_3',
                            'pent_vaccines_month_4',
                            'head_count_monthly_average_head_counts',
                            'general_outpatient_monthly_average_general_outpatient',
                            'deliveries_monthly_average_deliveries',
                            'pent_vaccines_monthly_average_pent_vaccines',
                            'ql_human_resources_score_human_resources',
                            'ql_human_resources_score_max_human_resources',
                            'ql_information_education_communication_score_information_education_communication',
                            'ql_information_education_communication_score_max_information_education_communication',
                            'ql_surveillance_score_surveillance', 'ql_surveillance_score_max_surveillance',
                            'ql_triage_and_early_recognition_score_triage_and_early_recognition',
                            'ql_triage_and_early_recognition_score_max_triage_and_early_recognition',
                            'ql_chw_score_chw',
                            'ql_chw_score_max_chw',
                            'ql_isolation_physical_distancing_score_isolation',
                            'ql_isolation_physical_distancing_score_max_isolation',
                            'ql_ppe_score_infection_prevention_and_control_ppe',
                            'ql_ppe_score_max_infection_prevention_and_control_ppe',
                            'ql_ppe_plan_score_infection_prevention_and_control_ppe_plan',
                            'ql_ppe_plan_score_max_infection_prevention_and_control_ppe_plan',
                            'ql_waste_collection_and_disposal_score_infection_prevention_and_control_waste_collection_and_disposal',
                            'ql_waste_collection_and_disposal_score_max_infection_prevention_and_control_waste_collection_and_disposal',
                            'ql_water_sanitation_and_hygiene_score_infection_prevention_and_control_water_sanitation_and_hygiene',
                            'ql_water_sanitation_and_hygiene_score_max_infection_prevention_and_control_water_sanitation_and_hygiene',
                            'ql_disinfection_and_sterilization_score_infection_prevention_and_control_disinfection_and_sterilization',
                            'ql_disinfection_and_sterilization_score_max_infection_prevention_and_control_disinfection_and_sterilization',
                            'grp_infection_prevention_and_control_score_infection_prevention_and_control',
                            'grp_infection_prevention_and_control_score_max_infection_prevention_and_control',
                            'question1_score_logistics_patient_and_sample_transfer',
                            'question1_score_max_logistics_patient_and_sample_transfer',
                            'score_total',
                            'score_max_total']

    # Get columns with missing values
    missing_df = df[missing_numeric_columns]

    # Initialize an instance of mice
    mice_imputer = IterativeImputer()
    filled_numerical_df = pd.DataFrame(mice_imputer.fit_transform(
        missing_df), columns=missing_numeric_columns)

    # Drop the missing values from the main dataframe
    df = df.drop(columns=missing_numeric_columns)
    df = df.drop(columns=['facility_gps'])

    # combine the dataframe with filled values with the main dataframe
    new_df = pd.concat([df, filled_numerical_df], axis=1)


    # FILL character columns with empty values with MODE
    missing_str_columns = ['ql_human_resources_hr_focal_point',
                        'ql_human_resources_hr_staff_received_info',
                        'ql_human_resources_hr_healthcare_provider_training',
                        'ql_human_resources_hr_healthcare_provider_revised_training',
                        'ql_human_resources_hr_daily_staff_list',
                        'ql_information_education_communication_iec_handwashing_procedure',
                        'ql_information_education_communication_iec_physical_distancing',
                        'ql_information_education_communication_iec_covering_nose_mouth',
                        'ql_information_education_communication_iec_early_symptom_recognition',
                        'ql_information_education_communication_iec_when_facility_vs_home',
                        'ql_information_education_communication_iec_rational_ppe_use',
                        'ql_information_education_communication_iec_helpline_number',
                        'ql_surveillance_surv_procedure_for_notification',
                        'ql_surveillance_surv_official_case_definition',
                        'ql_surveillance_surv_hotline_number',
                        'ql_surveillance_surv_timely_data_reported_to_district',
                        'ql_triage_and_early_recognition_ter_screening_area_set_up',
                        'ql_triage_and_early_recognition_ter_symptom_screening_questionnaires',
                        'ql_triage_and_early_recognition_ter_temperature_measurement_at_triage',
                        'ql_triage_and_early_recognition_ter_physical_distancing_in_waiting',
                        'ql_triage_and_early_recognition_ter_separate_waiting_for_symptomatic',
                        'ql_chw_chw_trained_precautions',
                        'ql_chw_chw_trained_community_service',
                        'ql_chw_chw_drugs',
                        'ql_chw_chw_gloves',
                        'ql_chw_chw_masks',
                        'ql_chw_chw_iec_materials',
                        'ql_isolation_physical_distancing_iso_designated_isolation_for_suspected',
                        'ql_isolation_physical_distancing_iso_distance_between_patients_in_waiting',
                        'ql_isolation_physical_distancing_iso_distance_between_patient_beds',
                        'ql_isolation_physical_distancing_iso_transfer_referral_protocol',
                        'ql_ppe_ipc_ppe_medical_masks',
                        'ql_ppe_ipc_ppe_disp_surgical_masks',
                        'ql_ppe_ipc_ppe_eye_protection',
                        'ql_ppe_ipc_ppe_examination_gloves',
                        'ql_ppe_ipc_ppe_surgical_gloves',
                        'ql_ppe_ipc_ppe_long_cuffed_gloves',
                        'ql_ppe_ipc_ppe_heavy_duty_gloves',
                        'ql_ppe_ipc_ppe_long_sleeved_gown',
                        'ql_ppe_ipc_ppe_waterproof_aprons',
                        'ql_ppe_plan_ipc_ppe_plan_staff_trained_on_ppe',
                        'ql_ppe_plan_ipc_ppe_plan_ppe_poster_displayed',
                        'ql_ppe_plan_ipc_ppe_plan_fit_test_kit',
                        'ql_ppe_plan_ipc_ppe_plan_contingency_for_shortages',
                        'ql_waste_collection_and_disposal_ipc_wcd_colour_coded_bins',
                        'ql_waste_collection_and_disposal_ipc_wcd_clinical_waste_bags',
                        'ql_waste_collection_and_disposal_ipc_wcd_laundry_receptacles_at_patient_rooms',
                        'ql_waste_collection_and_disposal_ipc_wcd_incinerator',
                        'ql_water_sanitation_and_hygiene_ipc_wash_clean_running_water',
                        'ql_water_sanitation_and_hygiene_ipc_wash_hand_soap',
                        'ql_water_sanitation_and_hygiene_ipc_wash_liquid_soap',
                        'ql_water_sanitation_and_hygiene_ipc_wash_disp_hand_towels',
                        'ql_water_sanitation_and_hygiene_ipc_wash_alcohol_based_hand_gel',
                        'ql_disinfection_and_sterilization_ipc_ds_protocol_facility_disinfection',
                        'ql_disinfection_and_sterilization_ipc_ds_protocol_equipment_sterilisation',
                        'ql_disinfection_and_sterilization_ipc_ds_environmental_disinfectant',
                        'ql_disinfection_and_sterilization_ipc_ds_cleaning_schedule_in_toilets',
                        'ql_disinfection_and_sterilization_ipc_ds_protocol_corpse_handling',
                        'ql_logistics_patient_and_sample_transfer_log_referral_plan',
                        'ql_logistics_patient_and_sample_transfer_log_cellphone_landline_swradio',
                        'ql_logistics_patient_and_sample_transfer_log_tracer_drugs',
                        'ql_logistics_patient_and_sample_transfer_log_albendazole',
                        'ql_logistics_patient_and_sample_transfer_log_amoxicillin',
                        'ql_logistics_patient_and_sample_transfer_log_ampicillin',
                        'ql_logistics_patient_and_sample_transfer_log_chlorhexidine_5',
                        'question10_log_chlorhexidine_7',
                        'question10_log_gentamicin',
                        'question10_log_folic',
                        'question10_log_ferrous_and_folic',
                        'question10_log_compound_sodium',
                        'question10_log_co_trimoxazole_400',
                        'question10_log_co_trimoxazole_200',
                        'question6_log_metronidazole_250',
                        'question6_log_metronidazole_200',
                        'question6_log_methyldopa',
                        'question6_log_magnesium_sulphate',
                        'question6_log_lidocaine',
                        'question6_log_ibuprofen',
                        'question2_log_surgical_spirit',
                        'question2_log_sodium_chloride',
                        'question2_log_povidone',
                        'question2_log_paracetamol_500',
                        'question2_log_paracetamol_250',
                        'question2_log_ors',
                        'question7_log_gauze',
                        'question7_log_cotton_wool',
                        'question7_log_cannula_iv_20',
                        'question7_log_cannula_iv_24',
                        'question7_log_zinc_sulphate',
                        'question7_log_water',
                        'question4_log_needle_23',
                        'question4_log_needle_21',
                        'question4_log_glove_giving',
                        'question4_log_glove_surgical',
                        'question4_log_glove_gyn',
                        'question4_log_glove_exam',
                        'question1_log_tape',
                        'question1_log_syringe',
                        'question1_log_oxytocin',
                        'question1_log_diazepam',
                        'question1_log_misoprostol',
                        'question1_log_glucose']

    df = impute_nan_add_vairable(new_df, missing_str_columns)
    df = df.dropna()


    # Dealing with Outliers
    #NB: This process can be ignored if loss of data is significant
    df = rectify_outliers_using_log_transformation(new_df, missing_numeric_columns)
    df = df.dropna()


    # Data Standardization
    scaler = StandardScaler()
    df[missing_numeric_columns] = scaler.fit_transform(df[missing_numeric_columns])

    # This is the final dataset
    return df

print(main())