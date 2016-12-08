from sklearn.preprocessing import FunctionTransformer

from reskit.norms import binar_norm, wbysqdist
from reskit.norms import spectral_norm

from reskit.features.degree import degrees 

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

import os
import pandas as pd
import numpy as np

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from reskit.core import Transformer, Pipeliner

def orig(x):
    return x

def get_autism(path_to_read='Data/dti/', distances=True):
    def get_autism_distances(loc_name):
        with open(loc_name, 'r') as f:
            read_data = f.readlines()

        read_data = pd.DataFrame(
            np.array([np.array(item[:-1].split()).astype(int) for item in read_data]))

        return read_data

    def get_distance_matrix(coords):
        if type(coords) == pd.core.frame.DataFrame:
            coords = coords.values
        elif type(coords) != np.ndarray:
            print('Provide either pandas df or numpy array!')
            return -1

        shape = len(coords)
        dist_matrix = np.zeros((shape, shape))
        del shape
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist_matrix[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    target_vector = []  # this will be a target vector (diagnosis)
    matrices = []  # this will be a list of connectomes
    all_files = sorted(os.listdir(path_to_read))
    matrix_files = [
        item for item in all_files if 'DTI_connectivity' in item and 'All' not in item]
    distance_files = [
        item for item in all_files if 'DTI_region_xyz_centers' in item and 'All' not in item]

    # for each file in a sorted (!) list of files:
    for filename in matrix_files:

        A_dataframe = pd.read_csv(
            path_to_read + filename, sep='   ', header=None, engine='python')
        A = A_dataframe.values  # we will use a list of numpy arrays, NOT pandas dataframes
        matrices.append(A)  # append a matrix to our list
        if "ASD" in filename:
            target_vector.append(1)
        elif "TD" in filename:
            target_vector.append(0)
    asd_dict = {}
    asd_dict['X'] = np.array(matrices)
    asd_dict['y'] = np.array(target_vector)
    if distances:
        dist_matrix_list = []
        for item in distance_files:
            # print(item)
            cur_coord = get_autism_distances(path_to_read + item)
            cur_dist_mtx = get_distance_matrix(cur_coord)
            dist_matrix_list += [cur_dist_mtx]

        asd_dict['dist'] = np.array(dist_matrix_list)

    return asd_dict

def get_baseline(path):
    values = pd.read_csv('Data/baseline/target.txt', header=None).values
    asd_dict = {}
    asd_dict['y'] = values.reshape(len(values)).astype('int')
    asd_dict['X'] = pd.read_csv('Data/baseline/final_baseline.csv').values
    return asd_dict

grid_cv = StratifiedKFold(n_splits=10,
                          shuffle=True,
                          random_state=0)

eval_cv = StratifiedKFold(n_splits=10,
                          shuffle=True,
                          random_state=1)

data = [('UCLAsource', Transformer(get_autism)),
        ('UCLAbaseline', Transformer(get_baseline))]

weighters = [('origW', Transformer(orig)),
             ('binar', Transformer(binar_norm)),
             ('wbysqdist', Transformer(wbysqdist))]

normalizers = [('origN', Transformer(orig)),
               ('spectral', Transformer(spectral_norm))]

featurizers = [('origF', Transformer(orig, collect=['X'])),
               ('degrees', Transformer(degrees, collect=['degrees']))]

selectors = [('var_threshold', VarianceThreshold())]

scalers = [('minmax', MinMaxScaler()),
           ('origS', FunctionTransformer(orig))]

classifiers = [('LR', LogisticRegression()),
               ('RF', RandomForestClassifier()),
               ('SVC', SVC()),
               ('XGB', XGBClassifier(nthread=1)),
               ('SGD', SGDClassifier())]

steps = [('Data', data),
         ('Weighters', weighters),
         ('Normalizers', normalizers),
         ('Featurizers', featurizers),
         ('Selectors', selectors),
         ('Scalers', scalers),
         ('Classifiers', classifiers)]

banned_combos = [('UCLAsource', 'origN'),
                 ('UCLAsource', 'origF'),
                 ('UCLAbaseline', 'degrees'),
                 ('UCLAbaseline', 'binar'),
                 ('UCLAbaseline', 'wbysqdist'),
                 ('UCLAbaseline', 'spectral'),
                 ('LR', 'origS'),
                 ('SVC', 'origS'),
                 ('SGD', 'origS'),
                 ('RF', 'minmax'),
                 ('XGB', 'minmax')]

param_grid = dict(
    LR=dict(
#        C=[0.01, 0.05, 0.1] + [0.05*i for i in range(3, 21)],
#        max_iter=[50, 100, 500],
        penalty=['l1', 'l2']
    ),
    SGD=dict(
#        alpha=[0.001, 0.01, 0.1, 0.5, 1.0],
#        l1_ratio=[0, 0.2, 0.4, 0.6, 0.8, 1],
#        loss=['hinge', 'log', 'modified_huber'],
        n_iter=[50, 100, 200],
#        penalty=['elasticnet']
    ),
    SVC=dict(
#        C=[0.0005, 0.001, 0.005, 0.01] + [i*0.05 for i in range(1,11)],
#        degree=[2, 3, 4],
#        kernel=['linear', 'poly', 'rbf', 'sigmoid'],
        max_iter=[50, 100, 150],
    ),
    RF=dict(
#        criterion=['entropy', 'gini'],
#        max_depth=[3, 5, 7, 10, 20],
#        max_features=['log2', 'sqrt'] + [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        n_estimators=[10, 50, 100, 200, 500]
    ),
    XGB=dict(
#        colsample_bytree=[0.05*i for i in range(1,21)],
#        learning_rate=[0.01*i for i in range(1,6)] + [0.05*i for i in range(2,11)],
#        max_depth=[i for i in range(1,12)],
#        n_estimators=[10, 50, 100, 200, 500],
#        nthread=[1],
#        reg_alpha=[0, 1],
#        reg_lambda=[0, 1],
        subsample=[0.5, 0.7, 1]
    )
)

pipe = Pipeliner(steps, eval_cv=eval_cv, grid_cv=grid_cv, param_grid=param_grid, banned_combos=banned_combos)
results = pipe.get_results('Data/dti/', caching_steps=['Data', 'Weighters', 'Normalizers', 'Featurizers'], scoring=['roc_auc'])
