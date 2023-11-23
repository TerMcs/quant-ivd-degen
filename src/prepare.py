import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import sys
import yaml

from dvclive import Live


def get_ids(file_path):
    """Data was originally split using R script. To preserve the split they are stored in .txt"""
    lines_list = []

    with open(file_path, 'r') as file:
        lines_list = [line.strip() for line in file]
    
    lines_list = [line.replace('"', '') for line in lines_list]
    
    return lines_list


def filter_by_icc(icc_results, icc_cut_off, feature_set, participant_info):
    
    icc_df = pd.read_csv(icc_results)
    icc_df = icc_df[icc_df['Type'] == "ICC2"]
    icc_df = icc_df[icc_df['ICC'] > icc_cut_off]
    features = icc_df['radiomic_feature'].tolist()
    
    if feature_set == "radiomics_dhi_dpsi":
        additional_features = ["ivd_height_index", "normalised_delta_si"]
    elif feature_set== "radiomics_only":
        additional_features = []
    elif feature_set == "dhi_dpsi_only":
        features = []
        additional_features = ["ivd_height_index", "normalised_delta_si"]
    else:
        sys.exit("Failed: feature_set not defined")
    
    features = features + participant_info + additional_features

    return features


def prepare_data(data, resampling, binwidth, features):

    df = data[(data["resampling"] == resampling) & (data["binwidth"] == binwidth)]    
    df = df.drop_duplicates(subset=['project_ID', 'level'])

    df = df[df['pfirrmann'] != '?'] # handful of levels with missing Pfirrmann grades removed
    df['pfirrmann'] = df['pfirrmann'].astype(float).astype(int)
    df = df[features]

    return df


def variance_thresholding(df, threshold, participant_info):

    columns_to_preserve = [col for col in df.columns if col in participant_info]
    columns_to_remove = [col for col in df.columns if col not in participant_info]
    data_to_process = df[columns_to_remove]

    # # Optionally scale the data before applying the variance threshold
    # scaler = StandardScaler()
    # # Fit and transform the scaler on your feature data
    # data_to_process = scaler.fit_transform(data_to_process)
    # Initialize VarianceThreshold with your chosen threshold (e.g., 0.01)
    variance_selector = VarianceThreshold(threshold)
    # Fit and transform the data to remove low-variance features
    data_processed = variance_selector.fit_transform(data_to_process)

    # Get the indices of the selected features
    selected_feature_indices = variance_selector.get_support(indices=True)
    removed_feature_names = [columns_to_remove[i] for i in range(len(columns_to_remove)) if i not in selected_feature_indices]
    df_filtered = df[columns_to_preserve + [columns_to_remove[i] for i in selected_feature_indices]]

    print("Low variance features removed:")
    print(len(removed_feature_names))
    print(removed_feature_names)

    return df_filtered


def correlation_analysis(live, df, participant_info, correlation_threshold):
    
    feature_df = df.drop(columns=participant_info)

    correlation_matrix = feature_df.corr().abs()

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_df)
    feature_df = pd.DataFrame(data=scaled_values, columns=feature_df.columns)

    fig, axes = plt.subplots(dpi=100)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=axes)
    live.log_image("original_correlation_matrix.png", fig)
    
    mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    correlation_matrix = correlation_matrix.mask(mask)
    
    highly_correlated_pairs = np.where(correlation_matrix > correlation_threshold)

    selected_features = []

    for i, j in zip(*highly_correlated_pairs):
        feature_i = feature_df.columns[i]
        feature_j = feature_df.columns[j]

        if feature_i == feature_j: # skip the correlation of feature with itself
            continue

        # check if the feature_i or feature_j are already in the selected features list 
        if feature_i in selected_features or feature_j in selected_features:
            continue
        
        # Calculate variances of the two features for making a choice which to keep
        variance_i = feature_df[feature_i].var()
        variance_j = feature_df[feature_j].var()
        selected_feature = feature_i if variance_i < variance_j else feature_j

        # ALTERNATIVE: use mutual information for feature selection:
        # mutual_info_i = mutual_info_classif(feature_df[feature_i].values.reshape(-1, 1), df["pfirrmann"].values)[0]
        # mutual_info_j = mutual_info_classif(feature_df[feature_j].values.reshape(-1, 1), df["pfirrmann"].values)[0]
        # selected_feature = feature_i if mutual_info_i < mutual_info_j else feature_j
        
        selected_features.append(selected_feature)

    df = df.drop(columns=selected_features)

    return df

# def chi_2_feature_selection(df, participant_info):
    
#     feature_df = df.drop(columns=participant_info)

#     fs = SelectKBest(score_func=chi2, k='all')


#     return df

def plot_correlation_matrix(live, df, participant_info):
    
    df = df.drop(columns=participant_info)

    correlation_matrix = df.corr().abs()

    # plot the updated correlation matrix and keep as experiment artifact
    fig, axes = plt.subplots(dpi=100)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", ax=axes)
    live.log_image("updated_correlation_matrix.png", fig)


def main():

    participant_info = ["project_ID", "level", "pfirrmann",  "C6646Q1_98_6_2", "gender", "C6631T_BMI", "C6646C_weigth_height_007", "C6646Q1_40_1"]

    params = yaml.safe_load(open("params.yaml"))["data"]

    # The main dataframe has been stored as a pickle file 
    with open(params['data_file'], 'rb') as f:
        data = pickle.load(f)

    features = filter_by_icc(params['icc_results'], 
                                params['icc_cut_off'], 
                                params['feature_set'],
                                participant_info
                                )
    df = prepare_data(data, 
                        params['resampling'],
                        params['binwidth'], 
                        features
                        )

    if params['split']:
        # split off the dev set before further processing to avoid data leakage
        dev = df[df['project_ID'].isin(get_ids(params['dev_ids']))]
    else:
        print('************* NOTE: no split created, dev_set_prepared.pkl will contain all participants. **********')
        dev = df

    dev = variance_thresholding(dev, 
                                params['variance_threshold'],
                                participant_info
                                )
   
   
    with Live(save_dvc_exp=True) as live:

        if params['correlation_threshold'] != 0:
            dev = correlation_analysis(live, 
                                      dev,
                                      participant_info,
                                      params['correlation_threshold']
                                      )


        plot_correlation_matrix(live, 
                                dev,
                                participant_info
                                )

    with open('data/dev_set_prepared.pkl', 'wb') as f:
        pickle.dump(dev, f)

    test = df[df['project_ID'].isin(get_ids(params['test_ids']))]

    # match the columns in case some were removed in the dev set preparation and won't be used in testing
    dev_columns = dev.columns.tolist()
    test = test[dev_columns]

    # save the dataframe:
    with open('data/test_set_prepared.pkl', 'wb') as f:
        pickle.dump(test, f)

if __name__ == "__main__":
    main()
