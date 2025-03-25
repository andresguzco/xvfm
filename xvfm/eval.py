import os
import json
import torch
import pandas as pd

from xvfm.mle import get_evaluator
from torch.nn.functional import one_hot
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import LogisticDetection
from synthcity.metrics import eval_statistical
from synthcity.plugins.core.dataloader import GenericDataLoader


TYPES = {
    "float32": "numerical",
    "float64": "numerical",
    "int32": "categorical",
    "int64": "categorical",
}


def get_mle(train, test, data_path):
    with open(os.path.join(data_path, "info.json"), "r") as f:
        info = json.load(f)

    task_type = info["task_type"]
    evaluator = get_evaluator(task_type)

    if task_type == "regression":
        r2, rmse = evaluator(train, test, info)
        overall_scores = {}
        for score_name in ["r2", "rmse"]:
            overall_scores[score_name] = {}
            scores = eval(score_name)
            for method in scores:
                name = method["name"]
                method.pop("name")
                overall_scores[score_name][name] = method
    else:
        print(f"Preparing MLE results")
        f1, auroc, acc, avg = evaluator(train, test, info)
        print(f"f1: {f1}, auroc: {auroc}, acc: {acc}, avg: {avg}")
        overall_scores = {}
        for score_name in ["f1", "auroc", "acc", "avg"]:
            overall_scores[score_name] = {}
            scores = eval(score_name)
            for method in scores:
                name = method["name"]
                method.pop("name")
                overall_scores[score_name][name] = method
    return overall_scores


def get_results(data):
    print(f"Data: {data}")
    max_metrics = {}
    for i, value in data.items():
        print(f"i: {i}, value: {value}")
        model_data = data[i]

        for metric, metric_value in model_data.items():
            print(f"metric: {metric}, metric_value: {metric_value}")
            if metric not in max_metrics:
                max_metrics[metric] = metric_value
            else:
                max_metrics[metric] = max(max_metrics[metric], metric_value)

    return max_metrics


def get_performance(syn_data, test_data, args, mle=False):
    num = cum_sum = args.num_feat
    k = sum(args.classes)

    syn_oh = torch.zeros(syn_data.shape[0], num + k + 1)
    test_oh = torch.zeros(syn_data.shape[0], num + k + 1)

    syn_oh[:, :num] = syn_data[:, :num]
    test_oh[:, :num] = test_data[:, :num]

    if sum(args.classes):
        for i, val in enumerate(args.classes):
            syn_oh[:, cum_sum : cum_sum + val] = one_hot(
                syn_data[:, num + i].to(torch.int64), num_classes=val
            )
            test_oh[:, cum_sum : cum_sum + val] = one_hot(
                test_data[:, num + i].to(torch.int64), num_classes=val
            )
            cum_sum += val

    syn_oh[:, -1] = syn_data[:, -1]
    test_oh[:, -1] = test_data[:, -1]

    syn_np = syn_data.cpu().numpy()
    test_np = test_data.cpu().numpy()

    syn = pd.DataFrame(syn_np)
    test = pd.DataFrame(test_np)

    shape, trend, quality = get_shape_trend_score(test, syn)
    detection = get_detection(test, syn)
    alpha, beta = get_quality(test_oh.cpu().numpy(), syn_oh.cpu().numpy())

    if mle:
        mle_scores = get_results(get_mle(syn_np, test_np, args.data_path))
    else:
        mle_scores = {}

    scores = {
        "shape": shape,
        "trend": trend,
        "detection": detection,
        "quality": quality,
        "alpha": alpha,
        "beta": beta,
    } | mle_scores

    return scores


def get_shape_trend_score(real, synthetic):
    metadata = {"primary_key": "user_id"}
    metadata["columns"] = {
        i: {"sdtype": TYPES[real.iloc[:, i].dtype.name]} for i in range(real.shape[1])
    }

    qual_report = QualityReport()
    qual_report.generate(real, synthetic, metadata, verbose=False)

    quality = qual_report.get_properties()

    Shape = quality["Score"][0]
    Trend = quality["Score"][1]
    Quality = (Shape + Trend) / 2

    return Shape, Trend, Quality


def get_detection(real, synthetic):

    for i in range(real.shape[1]):
        real.rename(columns={i: f"feat_{i}"}, inplace=True)
        synthetic.rename(columns={i: f"feat_{i}"}, inplace=True)

    TYPES = {
        "float32": "numerical",
        "float64": "numerical",
        "int32": "categorical",
        "int64": "categorical",
    }

    metadata = {"primary_key": "user_id"}
    metadata["columns"] = {
        f"feat_{i}": {"sdtype": TYPES[real.iloc[:, i].dtype.name]}
        for i in range(real.shape[1])
    }

    score = LogisticDetection.compute(
        real_data=real, synthetic_data=synthetic, metadata=metadata
    )
    return score


def get_quality(real_x, syn_x):
    sloader = GenericDataLoader(syn_x)
    rloader = GenericDataLoader(real_x)

    quality_evaluator = eval_statistical.AlphaPrecision()
    qual_res = quality_evaluator.evaluate(rloader, sloader)
    qual_res = {k: v for (k, v) in qual_res.items() if "naive" in k}

    Alpha = qual_res["delta_precision_alpha_naive"]
    Beta = qual_res["delta_coverage_beta_naive"]
    return Alpha, Beta
