import os
import json
import torch
import argparse
import pandas as pd
import numpy as np

from xvfm.mle import get_evaluator, get_results
from data.utils import Transformations, make_dataset, prepare_fast_dataloader
from torch.nn.functional import one_hot
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader


TYPES = {
    'float32': 'numerical',
    'float64': 'numerical',
    'int32': 'categorical',
    'int64': 'categorical'
}


def get_mle(train, test, data_path):
    with open(os.path.join(data_path, 'info.json'), 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    evaluator = get_evaluator(task_type)

    if task_type == 'regression':
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)
        overall_scores = {}
        for score_name in ['best_r2_scores', 'best_rmse_scores']:
            overall_scores[score_name] = {}
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    else:
        best_f1_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)
        overall_scores = {}
        for score_name in ['best_f1_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    return overall_scores


def get_performance(syn_data, test_data, args, mle=False):
    num = cum_sum = args.num_feat
    k = sum(args.classes)

    syn_oh = torch.zeros(syn_data.shape[0], num + k)
    test_oh = torch.zeros(syn_data.shape[0], num + k)

    syn_oh[:, :num] = syn_data[:, :num]
    test_oh[:, :num] = test_data[:, :num]

    if sum(args.classes):
        for i, val in enumerate(args.classes):
            syn_oh[:, cum_sum:cum_sum + val] = one_hot(syn_data[:, num+i].to(torch.int64), num_classes=val)
            test_oh[:, cum_sum:cum_sum + val] = one_hot(test_data[:, num+i].to(torch.int64), num_classes=val)
            cum_sum += val 

    syn_oh[:, -1] = syn_data[:, -1]
    test_oh[:, -1] = test_data[:, -1]

    test_np = test_data.cpu().numpy()
    syn_np = syn_data.cpu().numpy()

    test = pd.DataFrame(test_np)
    syn = pd.DataFrame(syn_np)

    max_shape, max_trend, avg_shape, avg_trend, shape, trend, quality = get_shape_trend_score(test, syn)
    detection = get_detection(test, syn)
    alpha, beta, qual_score = get_quality(test_oh.cpu().numpy(), syn_oh.cpu().numpy())
    
    if mle:
        mle_scores = get_results(get_mle(syn_np, test_np, args.data_path))
    else:
        mle_scores = {}

    scores = {
        "shape": shape,
        "max_shape": max_shape,
        "avg_shape": avg_shape,
        "trend": trend,
        "max_trend": max_trend,
        "avg_trend": avg_trend,
        "detection": detection,
        "quality": quality,
        "alpha": alpha,
        "beta": beta,
        "qual_score": qual_score
    } | mle_scores

    return scores


def get_shape_trend_score(real, synthetic):
    metadata = {'primary_key': 'user_id'}
    metadata['columns'] = {i: {'sdtype': TYPES[real.iloc[:, i].dtype.name]} for i in range(real.shape[1])}
        
    qual_report = QualityReport()
    qual_report.generate(real, synthetic, metadata, verbose=False)

    quality =  qual_report.get_properties()

    Shape = quality['Score'][0]
    Trend = quality['Score'][1]
    Quality = (Shape + Trend) / 2

    shapes = qual_report.get_details(property_name='Column Shapes')
    max_shape = np.max(shapes['Score'].values)
    avg_shape = np.mean(shapes['Score'].values)
    trends = qual_report.get_details(property_name='Column Pair Trends')
    max_trend = np.max(trends['Score'].values)
    avg_trend = np.mean(trends['Score'].values)
    return max_shape, max_trend, avg_shape, avg_trend, Shape, Trend, Quality


def get_detection(real, synthetic):

    for i in range(real.shape[1]):
        real.rename(columns={i: f'feat_{i}'}, inplace=True)
        synthetic.rename(columns={i: f'feat_{i}'}, inplace=True)

    TYPES = {
        'float32': 'numerical',
        'float64': 'numerical',
        'int32': 'categorical',
        'int64': 'categorical'
    }

    metadata = {'primary_key': 'user_id'}
    metadata['columns'] = {f'feat_{i}': {'sdtype': TYPES[real.iloc[:, i].dtype.name]} for i in range(real.shape[1])}

    score = LogisticDetection.compute(
        real_data=real,
        synthetic_data=synthetic,
        metadata=metadata
    )
    return score


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='TabVFM: Experiment')
    parser.add_argument('--batch_size', default=4096, type=int, help="Training batch size")
    parser.add_argument('--dataset', default='adult', type=str, help="Dataset to train the model on")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument('--results_dir', default='results', type=str, help="Directory to save results")
    parser.add_argument('--data_path', default=None, type=str, help="Path to tabular dataset")
    parser.add_argument('--num_classes', default=None, type=int, help="List of classes for tabular dataset")
    parser.add_argument('--transformation', default=None, type=str, help="Transformation to apply to tabular dataset")
    parser.add_argument('--is_y_cond', default=0, type=int, help="Flag to condition on y in tabular dataset")
    return parser.parse_args()


def get_quality(real_x, syn_x):
    sloader = GenericDataLoader(syn_x)
    rloader = GenericDataLoader(real_x)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(rloader, sloader)
    qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}
    qual_score = np.mean(list(qual_res.values()))

    Alpha = qual_res['delta_precision_alpha_naive']
    Beta = qual_res['delta_coverage_beta_naive']
    return Alpha, Beta, qual_score