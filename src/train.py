import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin 
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import yaml

from dvclive import Live
    

def load_data(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    participant_info = ['project_ID', 'level', 'pfirrmann', 'C6646Q1_98_6_2', 'gender', 'C6631T_BMI', 'C6646C_weigth_height_007', 'C6646Q1_40_1']

    X = data.drop(participant_info, axis=1)
    y = data['pfirrmann']

    return X, y

    
class PLSSelector(TransformerMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components, scale=False, max_iter=500, tol=1e-06) 

    def fit(self, X, y):
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_pls = self.pls.transform(X)  
        return X_pls


def make_pipeline(params):

    if params['model'] == 'pca_svm':
        scaler = StandardScaler()
        pca = PCA(n_components=params['n_components'])
        lr = LogisticRegression(multi_class='ovr', max_iter=1000)
        svm = SVC(kernel='linear', C=params['cost'], gamma=params['svm_gamma'], random_state=42)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('pca', pca),
            ('classifier', svm),
            # ('classifier', lr)
            ])
        
    elif params['model'] == 'svm':
        scaler = StandardScaler()
        svm = SVC(kernel='linear', C=params['cost'], gamma=params['svm_gamma'], random_state=42)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', svm)
            ])
        
    elif params['model'] == 'pls':
        scaler = StandardScaler()
        pls = PLSSelector(n_components=params['n_components'])
        lr = LogisticRegression(multi_class='ovr', max_iter=1000)
        svm = SVC(kernel='linear', C=params['cost'], gamma=params['svm_gamma'], random_state=42)
        random_forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('pls', pls),
            ('classifier', svm),
            # ('classifier', lr),
            # ('classifier', random_forest)
            ])

    
    elif params['model'] == 'lr':
        scaler = StandardScaler()
        lr = LogisticRegression(multi_class='ovr', max_iter=1000, C=params['cost'], random_state=42)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', lr)
            ])
    
    else:
        print("Model not defined")
        return
    
    return pipeline


def get_scorers():
    scorers = {
            'Accuracy': make_scorer(accuracy_score),
            'Macro F1 Score': make_scorer(f1_score, average='macro'),
            'Micro F1 Score': make_scorer(f1_score, average='micro'),
            'Balanced Accuracy Score': make_scorer(balanced_accuracy_score),
            'Matthews Correlation Coefficient': make_scorer(matthews_corrcoef),
            'Macro Precision': make_scorer(precision_score, average='macro'),
            'Micro Precision': make_scorer(precision_score, average='micro'),
            'Macro Recall': make_scorer(recall_score, average='macro'),
            'Micro Recall': make_scorer(recall_score, average='micro'),
            'Cohen Kappa': make_scorer(cohen_kappa_score)
            }
    return scorers


def prepare_y_for_pls(y):
    encoder = OneHotEncoder(sparse=False)
    # convert the target to one-hot encoding
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    y = np.argmax(y_encoded, axis=1)
    return y


def train_model(X, y, params):

    pipeline = make_pipeline(params)

    if params['model'] == 'pls':
        # y = prepare_y_for_pls(y)
        scorers = get_scorers()
        scores = cross_validate(pipeline, X, y, cv=5, scoring=scorers)
        scores.pop('fit_time')
        scores.pop('score_time')

    else:
        scorers = get_scorers()
        scores = cross_validate(pipeline, X, y, cv=5, scoring=scorers)
        scores.pop('fit_time')
        scores.pop('score_time')

    return scores


def main():

    params = yaml.safe_load(open("params.yaml"))["train"]

    X_train, y_train = load_data('data/dev_set_prepared.pkl')
    scores = train_model(X_train, y_train, params) 
    
    with Live(save_dvc_exp=True) as live: # in the development stage the results are added to dvclive so that they can be tracked
        for scorer_name, scorer_scores in scores.items():
            live.log_metric(f"{scorer_name}", scorer_scores.mean())

if __name__ == "__main__":
    main()







