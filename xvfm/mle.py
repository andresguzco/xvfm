import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    explained_variance_score, 
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score
    )
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid
from sklearn.utils._testing import ignore_warnings


CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
_MODELS = {
    'binclass': [
        {
            'class': XGBClassifier,
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 5, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 0.5, 1.0],
                 'objective': ['binary:logistic'],
                 'nthread': [-1],
                 'device': ['cpu'],
                 'tree_method': ['hist'],
                 'verbose': [0]
            },
        }

    ],
    'regression': [
        {
            'class': XGBRegressor,
            'kwargs': {
                 'n_estimators': [10, 50, 100],
                 'min_child_weight': [1, 5, 10], 
                 'max_depth': [5, 10, 20],
                 'gamma': [0.0, 0.5, 1.0],
                 'objective': ['reg:linear'],
                 'nthread': [-1],
                 'device': ['cpu'],
                 'tree_method': ['hist'],
                 'verbose': [0]
            }
        },
    ]
}


def get_results(data):
    max_metrics = {}
    for key, value in data.items():
        model_data = next(iter(value.values()), {})

        for metric, metric_value in model_data.items():
            if metric not in max_metrics:
                max_metrics[metric] = metric_value
            else:
                max_metrics[metric] = max(max_metrics[metric], metric_value)
    return max_metrics


def feat_transform(data, info, cmax = None, cmin = None):
    num_feat = len(info['num_col_idx'])

    features = []
    
    for idx in range(num_feat):
        col = data[:, idx]
        col = col.astype(np.float32)

        if not cmin:
            cmin = col.min()
        if not cmax:
            cmax = col.max()
        if cmin >= 0 and cmax >= 1e3:
            feature = np.log(np.maximum(col, 1e-2))
        else:
            feature = (col - cmin) / (cmax - cmin) * 5

        features.append(feature)

    features = np.column_stack(features)
    return features, data[:, -1], cmax, cmin


def prepare_ml_problem(train, test, info):
    train_X, train_y, cmax, cmin = feat_transform(train, info)
    test_X, test_y, _, _ = feat_transform(test, info, cmax, cmin)

    total_train_num = train_X.shape[0]
    val_num = int(total_train_num / 9)

    total_train_idx = np.arange(total_train_num)
    np.random.shuffle(total_train_idx)
    train_idx = total_train_idx[val_num:]
    val_idx = total_train_idx[:val_num]

    val_X, val_y = train_X[val_idx], train_y[val_idx]
    train_X, train_y = train_X[train_idx], train_y[train_idx]
    
    model = _MODELS[info['task_type']]
    
    return train_X, train_y, val_X, val_y, test_X, test_y, model


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification(train, test, info):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = prepare_ml_problem(train, test, info)

    unique_labels = np.unique(y_trains)

    best_f1_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []

    for model_spec in classifiers:

        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains)

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in param_set:
            model = model_class(**param)
            model.fit(x_trains, y_trains)

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_valid)
                pred_prob = np.array([1.] * len(x_valid))
            else:
                pred = model.predict(x_valid)
                pred_prob = model.predict_proba(x_valid)

            binary_f1 = f1_score(y_valid, pred, average='binary')
            acc = accuracy_score(y_valid, pred)
            precision = precision_score(y_valid, pred, average='binary')
            recall = recall_score(y_valid, pred, average='binary')
            macro_f1 = f1_score(y_valid, pred, average='macro')

            # auroc
            size = 2
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valid.shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1

            roc_auc = roc_auc_score(np.eye(size)[y_valid.astype(int)], np.hstack(tmp))

            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "binary_f1": binary_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc, 
                    "precision": precision, 
                    "recall": recall, 
                    "macro_f1": macro_f1
                }
            )


        # test the best model
        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['binary_f1', 'roc_auc']].mean(axis=1)  
        try:      
            best_f1_param = results.param[results.binary_f1.idxmax()]
        except: 
            best_f1_param = {
                 'n_estimators': 50,
                 'min_child_weight': 3, 
                 'max_depth': 5,
                 'gamma': 0.5,
                 'objective': 'binary:logistic',
                 'nthread': -1,
                 'device': 'cpu',
                 'tree_method': 'hist',
                 'verbose': 0
                }

        try:      
            best_auroc_param = results.param[results.roc_auc.idxmax()]
        except: 
            best_auroc_param = {
                 'n_estimators': 50,
                 'min_child_weight': 3, 
                 'max_depth': 5,
                 'gamma': 0.5,
                 'objective': 'binary:logistic',
                 'nthread': -1,
                 'device': 'cpu',
                 'tree_method': 'hist',
                 'verbose': 0
                }

        try:      
            best_acc_param = results.param[results.accuracy.idxmax()]
        except: 
            best_acc_param = {
                 'n_estimators': 50,
                 'min_child_weight': 3, 
                 'max_depth': 5,
                 'gamma': 0.5,
                 'objective': 'binary:logistic',
                 'nthread': -1,
                 'device': 'cpu',
                 'tree_method': 'hist',
                 'verbose': 0
                }

        try:      
            best_avg_param = results.param[results.avg.idxmax()]
        except: 
            best_avg_param = {
                 'n_estimators': 50,
                 'min_child_weight': 3, 
                 'max_depth': 5,
                 'gamma': 0.5,
                 'objective': 'binary:logistic',
                 'nthread': -1,
                 'device': 'cpu',
                 'tree_method': 'hist',
                 'verbose': 0
                }

        def _calc(best_model):
            best_scores = []

            best_model.fit(x_trains, y_trains)

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_test)
                pred_prob = np.array([1.] * len(x_test))
            else:
                pred = best_model.predict(x_test)
                pred_prob = best_model.predict_proba(x_test)

            binary_f1 = f1_score(y_test, pred, average='binary')
            acc = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='binary')
            recall = recall_score(y_test, pred, average='binary')
            macro_f1 = f1_score(y_test, pred, average='macro')

            # auroc
            size = 2
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:,[j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            try:
                roc_auc = roc_auc_score(np.eye(size)[y_test.astype(int)], np.hstack(tmp))
            except ValueError:
                tmp[1] = tmp[1].reshape(20000, 1)
                roc_auc = roc_auc_score(np.eye(size)[y_test.astype(int)], np.hstack(tmp))

            best_scores.append(
                {   
                    "name": model_repr,
                    # "param": param,
                    "binary_f1": binary_f1,
                    "roc_auc": roc_auc, 
                    "accuracy": acc, 
                    "precision": precision, 
                    "recall": recall, 
                    "macro_f1": macro_f1
                }
            )

            return pd.DataFrame(best_scores)
        def _df(dataframe):
            return {
                "name": model_repr,
                "binary_f1": dataframe.binary_f1.values[0],
                "roc_auc": dataframe.roc_auc.values[0],
                "accuracy": dataframe.accuracy.values[0],
            }
        
        best_f1_scores.append(_df(_calc(model_class(**best_f1_param))))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param))))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param))))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param))))

    return best_f1_scores, best_auroc_scores, best_acc_scores, best_avg_scores


@ignore_warnings(category=ConvergenceWarning)
def _evaluate_regression(train, test, info):
    
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, regressors = prepare_ml_problem(train, test, info)
    
    best_r2_scores = []
    best_ev_scores = []
    best_mae_scores = []
    best_rmse_scores = []

    y_trains = np.log(np.clip(y_trains, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))

    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in param_set:
            model = model_class(**param)
            model.fit(x_trains, y_trains)
            pred = model.predict(x_valid)

            r2 = r2_score(y_valid, pred)
            explained_variance = explained_variance_score(y_valid, pred)
            mean_squared = mean_squared_error(y_valid, pred)
            root_mean_squared = np.sqrt(mean_squared)
            mean_absolute = mean_absolute_error(y_valid, pred)

            results.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "r2": r2,
                    "explained_variance": explained_variance,
                    "mean_squared": mean_squared, 
                    "mean_absolute": mean_absolute, 
                    "rmse": root_mean_squared
                }
            )

        results = pd.DataFrame(results)
        best_r2_param = results.param[results.r2.idxmax()]
        best_ev_param = results.param[results.explained_variance.idxmax()]
        best_mae_param = results.param[results.mean_absolute.idxmin()]
        best_rmse_param = results.param[results.rmse.idxmin()]

        def _calc(best_model):
            best_scores = []
            x_train, y_train = x_trains, y_trains
            
            best_model.fit(x_train, y_train)
            pred = best_model.predict(x_test)

            r2 = r2_score(y_test, pred)
            explained_variance = explained_variance_score(y_test, pred)
            mean_squared = mean_squared_error(y_test, pred)
            root_mean_squared = np.sqrt(mean_squared)
            mean_absolute = mean_absolute_error(y_test, pred)

            best_scores.append(
                {   
                    "name": model_repr,
                    "param": param,
                    "r2": r2,
                    "explained_variance": explained_variance,
                    "mean_squared": mean_squared, 
                    "mean_absolute": mean_absolute, 
                    "rmse": root_mean_squared
                }
            )

            return pd.DataFrame(best_scores)

        def _df(dataframe):
            return {
                "name": model_repr,
                "r2": dataframe.r2.values[0].astype(float),
                "explained_variance": dataframe.explained_variance.values[0].astype(float),
                "MAE": dataframe.mean_absolute.values[0].astype(float),
                "RMSE": dataframe.rmse.values[0].astype(float),
            }

        best_r2_scores.append(_df(_calc(model_class(**best_r2_param))))
        best_ev_scores.append(_df(_calc(model_class(**best_ev_param))))
        best_mae_scores.append(_df(_calc(model_class(**best_mae_param))))
        best_rmse_scores.append(_df(_calc(model_class(**best_rmse_param))))

    return best_r2_scores, best_rmse_scores


_EVALUATORS = {
    'binclass': _evaluate_binary_classification,
    'regression': _evaluate_regression
}


def get_evaluator(problem_type):
    return _EVALUATORS[problem_type]
