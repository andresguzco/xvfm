import os
import enum
import json
import torch
import numpy as np
import pandas as pd
import pickle
import sklearn
import scipy.special
import sklearn.metrics as skm

from typing import Any, Literal, Optional, Union, cast, Tuple, Dict, List
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, replace
from sklearn.pipeline import make_pipeline
from category_encoders import LeaveOneOutEncoder


ArrayDict = Dict[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]


CAT_RARE_VALUE = "__rare__"
Normalization = Literal["standard", "quantile", "minmax"]
CatEncoding = Literal["one-hot", "counter"]
YPolicy = Literal["default"]


class TaskType(enum.Enum):
    BINCLASS = "binclass"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"

    def __str__(self) -> str:
        return self.value


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def read_pure_data(path, split="train"):
    y = np.load(os.path.join(path, f"y_{split}.npy"), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f"X_num_{split}.npy")):
        X_num = np.load(os.path.join(path, f"X_num_{split}.npy"), allow_pickle=True)
    if os.path.exists(os.path.join(path, f"X_cat_{split}.npy")):
        X_cat = np.load(os.path.join(path, f"X_cat_{split}.npy"), allow_pickle=True)

    return X_num, X_cat, y


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    @classmethod
    def from_dir(cls, dir_: Union[Path, str]) -> "Dataset":
        dir_ = Path(dir_)
        splits = [
            k for k in ["train", "val", "test"] if dir_.joinpath(f"y_{k}.npy").exists()
        ]

        def load(item) -> ArrayDict:
            return {
                x: cast(np.ndarray, np.load(dir_ / f"{item}_{x}.npy", allow_pickle=True))  # type: ignore[code]
                for x in splits
            }

        if Path(dir_ / "info.json").exists():
            info = load_json(dir_ / "info.json")
        else:
            info = None
        return Dataset(
            load("X_num") if dir_.joinpath("X_num_train.npy").exists() else None,
            load("X_cat") if dir_.joinpath("X_cat_train.npy").exists() else None,
            load("y"),
            {},
            TaskType(info["task_type"]),
            info.get("n_classes"),
        )

    @property
    def is_binclass(self) -> bool:
        return self.task_type == TaskType.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.task_type == TaskType.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.task_type == TaskType.REGRESSION

    @property
    def n_num_features(self) -> int:
        return 0 if self.X_num is None else self.X_num["train"].shape[1]

    @property
    def n_cat_features(self) -> int:
        return 0 if self.X_cat is None else self.X_cat["train"].shape[1]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: Optional[str]) -> int:
        return sum(map(len, self.y.values())) if part is None else len(self.y[part])

    @property
    def nn_output_dim(self) -> int:
        if self.is_multiclass:
            assert self.n_classes is not None
            return self.n_classes
        else:
            return 1

    def get_category_sizes(self, part: str) -> List[int]:
        return [] if self.X_cat is None else get_category_sizes(self.X_cat[part])

    def calculate_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        prediction_type: Optional[str],
    ) -> Dict[str, Any]:
        metrics = {
            x: calculate_metrics(
                self.y[x], predictions[x], self.task_type, prediction_type, self.y_info
            )
            for x in predictions
        }
        if self.task_type == TaskType.REGRESSION:
            score_key = "rmse"
            score_sign = -1
        else:
            score_key = "accuracy"
            score_sign = 1
        for part_metrics in metrics.values():
            part_metrics["score"] = score_sign * part_metrics[score_key]
        return metrics


class PredictionType(enum.Enum):
    LOGITS = "logits"
    PROBS = "probs"


class MetricsReport:
    def __init__(self, report: dict, task_type: TaskType):
        self._res = {k: {} for k in report.keys()}
        if task_type in (TaskType.BINCLASS, TaskType.MULTICLASS):
            self._metrics_names = ["acc", "f1"]
            for k in report.keys():
                self._res[k]["acc"] = report[k]["accuracy"]
                self._res[k]["f1"] = report[k]["macro avg"]["f1-score"]
                if task_type == TaskType.BINCLASS:
                    self._res[k]["roc_auc"] = report[k]["roc_auc"]
                    self._metrics_names.append("roc_auc")

        elif task_type == TaskType.REGRESSION:
            self._metrics_names = ["r2", "rmse"]
            for k in report.keys():
                self._res[k]["r2"] = report[k]["r2"]
                self._res[k]["rmse"] = report[k]["rmse"]
        else:
            raise "Unknown TaskType!"

    def get_splits_names(self) -> list[str]:
        return self._res.keys()

    def get_metrics_names(self) -> list[str]:
        return self._metrics_names

    def get_metric(self, split: str, metric: str) -> float:
        return self._res[split][metric]

    def get_val_score(self) -> float:
        return (
            self._res["val"]["r2"]
            if "r2" in self._res["val"]
            else self._res["val"]["f1"]
        )

    def get_test_score(self) -> float:
        return (
            self._res["test"]["r2"]
            if "r2" in self._res["test"]
            else self._res["test"]["f1"]
        )

    def print_metrics(self) -> None:
        res = {
            "val": {k: np.around(self._res["val"][k], 4) for k in self._res["val"]},
            "test": {k: np.around(self._res["test"][k], 4) for k in self._res["test"]},
        }

        print("*" * 100)
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])

        return res


class SeedsMetricsReport:
    def __init__(self):
        self._reports = []

    def add_report(self, report: MetricsReport) -> None:
        self._reports.append(report)

    def get_mean_std(self) -> dict:
        res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                res[split][metric] = [
                    x.get_metric(split, metric) for x in self._reports
                ]

        agg_res = {k: {} for k in ["train", "val", "test"]}
        for split in self._reports[0].get_splits_names():
            for metric in self._reports[0].get_metrics_names():
                for k, f in [("count", len), ("mean", np.mean), ("std", np.std)]:
                    agg_res[split][f"{metric}-{k}"] = f(res[split][metric])
        self._res = res
        self._agg_res = agg_res

        return agg_res

    def print_result(self) -> dict:
        res = {
            split: {
                k: float(np.around(self._agg_res[split][k], 4))
                for k in self._agg_res[split]
            }
            for split in ["val", "test"]
        }
        print("=" * 100)
        print("EVAL RESULTS:")
        print("[val]")
        print(res["val"])
        print("[test]")
        print(res["test"])
        print("=" * 100)
        return res


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = "default"


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(data_path: str, num_classes: int):
    T = Transformations(normalization="quantile")

    # classification
    if num_classes > 0:
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {}

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {}

        for split in ["train", "test"]:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    info = load_json(os.path.join(data_path, "info.json"))

    D = Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=TaskType(info["task_type"]),
        n_classes=info.get("n_classes"),
    )

    return transform_dataset(D, T)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f"Unknown {unknown_what}: {unknown_value}")


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def normalize(
    X,
    normalization: Normalization,
    seed: Optional[int],
    return_normalizer: bool = False,
):
    X_train = X["train"].astype("int64")
    if normalization == "standard":
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == "minmax":
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == "quantile":
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(X["train"].shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise_unknown("normalization", normalization)
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


def cat_encode(
    X: ArrayDict,
    encoding: Optional[CatEncoding],
    y_train: Optional[np.ndarray],
    seed: Optional[int],
    return_encoder: bool = False,
) -> Tuple[ArrayDict, bool, Optional[Any]]:  # (X, is_converted_to_numerical)
    if encoding != "counter":
        y_train = None

    # Step 1. Map strings to 0-based ranges

    if encoding is None:
        unknown_value = np.iinfo("int64").max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value",  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype="int64",  # type: ignore[code]
        ).fit(X["train"])
        encoder = make_pipeline(oe)
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X["train"].max(axis=0)
        for part in X.keys():
            if part == "train":
                continue
            for column_idx in range(X[part].shape[1]):
                X[part][X[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )
        if return_encoder:
            return (X, False, encoder)
        return (X, False)

    # Step 2. Encode.

    elif encoding == "one-hot":
        ohe = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse=False, dtype=np.float32  # type: ignore[code]
        )
        encoder = make_pipeline(ohe)

        # encoder.steps.append(('ohe', ohe))
        encoder.fit(X["train"])
        X = {k: encoder.transform(v) for k, v in X.items()}
    elif encoding == "counter":
        assert y_train is not None
        assert seed is not None
        loe = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
        encoder.steps.append(("loe", loe))
        encoder.fit(X["train"], y_train)
        X = {k: encoder.transform(v).astype("float32") for k, v in X.items()}  # type: ignore[code]
        if not isinstance(X["train"], pd.DataFrame):
            X = {k: v.values for k, v in X.items()}  # type: ignore[code]
    else:
        util.raise_unknown("encoding", encoding)

    if return_encoder:
        return X, True, encoder  # type: ignore[code]
    return (X, True)


def transform_dataset(dataset: Dataset, transformations: Transformations) -> Dataset:
    # WARNING: the order of transformations matters. Moreover, the current
    # implementation is not ideal in that sense.

    num_transform = None
    cat_transform = None
    X_num = dataset.X_num

    if X_num is not None:
        X_num, num_transform = normalize(
            X_num,
            transformations.normalization,
            transformations.seed,
            return_normalizer=True,
        )
        num_transform = num_transform

    if dataset.X_cat is None or dataset.X_cat["train"].size == 0:
        assert transformations.cat_min_frequency is None
        X_cat = None
    else:
        X_cat = dataset.X_cat
        if transformations.cat_min_frequency is not None:
            X_cat = cat_drop_rare(X_cat, transformations.cat_min_frequency)
        X_cat, is_num, cat_transform = cat_encode(
            X_cat,
            transformations.cat_encoding,
            dataset.y["train"],
            transformations.seed,
            return_encoder=True,
        )
        if is_num:
            X_num = (
                X_cat
                if X_num is None
                else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
            )
            X_cat = None

    y, y_info = build_target(dataset.y, transformations.y_policy, dataset.task_type)

    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.cat_transform = cat_transform
    return dataset


def cat_drop_rare(X: ArrayDict, min_frequency: float) -> ArrayDict:
    assert 0.0 < min_frequency < 1.0
    min_count = round(len(X["train"]) * min_frequency)
    X_new = {x: [] for x in X}
    for column_idx in range(X["train"].shape[1]):
        counter = Counter(X["train"][:, column_idx].tolist())
        popular_categories = {k for k, v in counter.items() if v >= min_count}
        for part in X_new:
            X_new[part].append(
                [
                    (x if x in popular_categories else CAT_RARE_VALUE)
                    for x in X[part][:, column_idx].tolist()
                ]
            )
    return {k: np.array(v).T for k, v in X_new.items()}


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def build_target(
    y: ArrayDict, policy: Optional[YPolicy], task_type: TaskType
) -> Tuple[ArrayDict, Dict[str, Any]]:
    info: Dict[str, Any] = {"policy": policy}
    if policy is None:
        pass
    elif policy == "default":
        if task_type == TaskType.REGRESSION:
            mean, std = float(y["train"].mean()), float(y["train"].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info["mean"] = mean
            info["std"] = std
    else:
        raise_unknown("policy", policy)
    return y, info


def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        util.raise_unknown("prediction_type", prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype("int64"), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert "std" in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info["std"])
        r2 = skm.r2_score(y_true, y_pred)
        result = {"rmse": rmse, "r2": r2}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result["roc_auc"] = skm.roc_auc_score(y_true, probs)
    return result


class FastTensorDataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=False, classes=None, num_feat=None):
        # assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.X = X
        self.y = y
        self.classes = classes
        self.num_feat = num_feat

        self.dataset_len = self.X.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.X = self.X[r]
            self.y = self.y[r]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = [
            self.X[self.i : self.i + self.batch_size],
            self.y[self.i : self.i + self.batch_size],
        ]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def prepare_test_data(D):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(
                np.concatenate([D.X_num["test"], D.X_cat["test"]], axis=1)
            ).float()
        else:
            X = torch.from_numpy(D.X_cat["test"]).float()
    else:
        X = torch.from_numpy(D.X_num["test"]).float()

    try:
        y = torch.from_numpy(D.y["test"])
    except:
        y = torch.from_numpy(np.where(D.y["test"], 1, 0))

    K = D.get_category_sizes("test")
    if len(K) == 0:
        K = [0]

    data = torch.cat([X, y.view(-1, 1)], dim=1)
    return data


def prepare_fast_dataloader(D: Dataset, split: str, batch_size: int):
    if D.X_cat is not None:
        if D.X_num is not None:
            X = torch.from_numpy(
                np.concatenate([D.X_num[split], D.X_cat[split]], axis=1)
            ).float()
        else:
            X = torch.from_numpy(D.X_cat[split]).float()
    else:
        X = torch.from_numpy(D.X_num[split]).float()

    try:
        y = torch.from_numpy(D.y[split])
    except:
        y = torch.from_numpy(np.where(D.y[split], 1, 0))

    K = D.get_category_sizes(split)
    if len(K) == 0:
        K = [0]

    nf = D.X_num["train"].shape[1] if D.X_num["train"] is not None else 0
    dataloader = FastTensorDataLoader(
        X, y, batch_size=batch_size, shuffle=(split == "train"), classes=K, num_feat=nf
    )

    return dataloader
