import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from tqdm import tqdm
import yaml

from dvclive import Live
    
from train import load_data, PLSSelector, make_pipeline, prepare_y_for_pls

def train_model(X, y, params):

    pipeline = make_pipeline(params)

    if params['model'] == 'pls':
        y = prepare_y_for_pls(y)
        pipeline.fit(X, y)
    else: 
        pipeline.fit(X, y)

    return pipeline


def bootstrap_test(metric, y, preds, null_hypothesis, n_bootstrap, seed=12345, stratified=True, alpha=95, metric_average=None):
    """
    Parameters
    ----------
    metric : fucntion
        Metric to compute, e.g. AUC for ROC curve or AP for PR curve
    y : numpy.array
        Ground truth
    preds : numpy.array
        Predictions
    null_hypothesis :
        Value for the metric if predictions are random or some other kind of null hypothesis for testing the model
        performance against
    n_bootstrap:
        Number of bootstrap samples to draw
    seed : int
        Random seed
    stratified : bool
        Whether to do a stratified bootstrapping
    alpha : float
        Confidence intervals width
    """

    np.random.seed(seed)
    metric_vals = []
    classes = np.unique(y)
    inds = []
    for cls in classes:
        inds.append(np.where(y == cls)[0])

    for _ in tqdm(range(n_bootstrap), total=n_bootstrap, desc='Bootstrap:'):
        if stratified:
            ind_bs = []
            for ind_cur in inds:
                ind_bs.append(np.random.choice(ind_cur, ind_cur.shape[0]))
            ind = np.hstack(ind_bs)
        else:
            ind = np.random.choice(y.shape[0], y.shape[0])

        if metric_average is not None:
            bootstrap_statistic = metric(y[ind], preds[ind], average=metric_average)
        else:
            bootstrap_statistic = metric(y[ind], preds[ind])

        if bootstrap_statistic is not None:
            metric_vals.append(bootstrap_statistic)

    metric_vals = np.array(metric_vals)
    p_value = np.mean(metric_vals <= null_hypothesis)   # I am still not confident on this 
                                                        # (e.g. this looks like one sided test, should it not be two-tailed? 

    if metric_average is not None:
        metric_val = metric(y, preds, average=metric_average)
    else:
        metric_val = metric(y, preds)

    ci_l = np.percentile(metric_vals, (100 - alpha) // 2)
    ci_h = np.percentile(metric_vals, alpha + (100 - alpha) // 2)

    return p_value, metric_val, ci_l, ci_h


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator / denominator


def test_model(X_test, y_test, pipeline, params):

    y_pred = pipeline.predict(X_test)

    if params['model'] == 'pls':
        y_pred = y_pred + 2

    feature_sets = ['radiomics_dhi_dpsi', 'radiomics_only', 'dhi_dpsi_only']

    dfs = []

    metrics_multiclass = {
            'Accuracy': accuracy_score,
            'Balanced Accuracy Score': balanced_accuracy_score,
            'Matthews Correlation Coefficient': matthews_corrcoef,
            'Cohen Kappa': cohen_kappa_score,
            'Lins Concordance Correlation Coefficient': concordance_correlation_coefficient,
            }

    metrics_averaged = {
            'F1 Score': f1_score,
            'Precision': precision_score, 
            'Recall': recall_score,
            }
    
    for feature_set in feature_sets:

        evaluation_results = {}

        for name, score in metrics_multiclass.items():
            print(f'Computing {name}')
            p_value, metric_val, ci_l, ci_h = bootstrap_test(score, y_test, y_pred, null_hypothesis=0.5, n_bootstrap=1000)
            evaluation_results[name] = [metric_val, ci_l, ci_h, p_value, feature_set]

        evaluation_results = pd.DataFrame.from_dict(evaluation_results, orient='index', columns=['Metric Value', 'CI Lower', 'CI Upper', 'p-value', 'Feature Set'])
        
        for name, score in metrics_averaged.items():
            for avg in ['macro', 'micro']:
                print(f'Computing {name} ({avg})')
                p_value, metric_val, ci_l, ci_h = bootstrap_test(score, y_test, y_pred, null_hypothesis=0.5, n_bootstrap=1000, metric_average=avg)
                evaluation_results.loc[f'{name} ({avg})'] = [metric_val, ci_l, ci_h, p_value, feature_set]
        
        dfs.append(evaluation_results)

    metrics_df = pd.concat(dfs, axis=0)

    return y_pred, metrics_df


def save_artifacts(y_test, y_pred, metrics_df, pipeline, params):

    # check if data/test_results_dir exists, if not create it:
    if not os.path.exists(f'data/{params["test_results_directory"]}'):
        os.makedirs(f'data/{params["test_results_directory"]}')

    # save the predictions for evaluations:
    predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    predictions_df.to_csv(f'data/{params["test_results_directory"]}/predictions.csv', index=False)

    metrics_df.to_csv(f'data/{params["test_results_directory"]}/metrics.csv')

    with open(f'data/{params["test_results_directory"]}/model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)


def main():

    params = yaml.safe_load(open("params.yaml"))["train"]

    X_train, y_train = load_data('data/dev_set_prepared.pkl')
    pipeline = train_model(X_train, y_train, params) # if params['test'] is True, the model will be trained on the whole dev dataset
    
    X_test, y_test = load_data('data/test_set_prepared.pkl')
    y_pred, metrics_df = test_model(X_test, y_test, pipeline, params) # this will save a csv file of the model predictions on the test set, which can be input for evaluate.py

    with Live(save_dvc_exp=True) as live: # in the development stage the results are added to dvclive so that they can be tracked
        live.log_metric('Accuracy', metrics_df.loc['Accuracy', 'Metric Value'])
        live.log_metric('Balanced Accuracy Score', metrics_df.loc['Balanced Accuracy Score', 'Metric Value'])
        live.log_metric('Cohen Kappa', metrics_df.loc['Cohen Kappa', 'Metric Value'])
        live.log_metric('Lins Concordance Correlation Coefficient', metrics_df.loc['Lins Concordance Correlation Coefficient', 'Metric Value'])

    params = yaml.safe_load(open("params.yaml"))["test"]

    if params['save_artifacts']:
        save_artifacts(y_test, y_pred, metrics_df, pipeline, params)

if __name__ == "__main__":
    main()







