import torch
import ignite
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fid import FIDNet

from ignite.metrics import FID
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage

def evaluate(args, model, savedir, device=None, suffix: str = None, xt: torch.Tensor = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    traj = model.generate(
        num_samples=1024 if args.dataset == 'two_moons' else 100 if args.dataset == 'mnist' else 4096, 
        steps=args.integration_steps, 
        device=device,
        method=args.int_method
    )

    if args.dataset in ['two_moons', 'mnist']:
        if args.dataset == 'two_moons':
            n = 2000
            traj = traj.cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
            plt.scatter(traj[:-2, :n, 0], traj[:-2, :n, 1], s=0.2, alpha=0.2, c="olive")
            plt.scatter(traj[-2, :n, 0], traj[-2, :n, 1], s=4, alpha=1, c="blue")
            plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
            plt.xticks([])
            plt.yticks([])   
            filename = f"{savedir}/iter_{suffix}.png" if suffix else f"{savedir}/final.png"
            plt.savefig(filename)
            plt.close()
            return None

        elif args.dataset == 'mnist':
            real_dataset = MNIST(
                root='data', train=False, download=True, 
                transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                )
            real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=100, shuffle=True)
            real_images = next(iter(real_loader))[0].to(device)[:100]
            generated_images = traj[-2, :100].view([-1, 1, 28, 28]).clip(-1, 1).to(device)

            evaluator = FIDNet()
            evaluator.load_state_dict(torch.load("checkpoints/fid_model.pt"))
            m = FID(num_features=10, device=device, feature_extractor=evaluator)
            m.update((real_images, generated_images))
            fid = m.compute()

            if plot:
                grid = make_grid(generated_images, value_range=(-1, 1), padding=0, nrow=10)
                img = ToPILImage()(grid)

                plt.imshow(img)
                plt.title(f"FID: {fid:.2f}")
                filename = f"{savedir}/iter_{suffix}.png" if suffix else f"{savedir}/final.png"
                plt.savefig(filename)
                plt.close()

            return fid
        else: pass
    else:
        k = model.variational_dist.num_feat + len(model.variational_dist.cat_feat)
        gen = torch.zeros(traj.shape[1], k)
        num = cum_sum = model.variational_dist.num_feat
        for i, val in enumerate(model.variational_dist.cat_feat):
            gen[:, num + i] = torch.argmax(traj[-2, :, cum_sum: cum_sum + val], dim=1)

        data = xt.cpu().numpy()
        synth = gen.numpy()

        # print(f"Data shape: {data.shape}, Synth shape: {synth.shape}")
        # print(f"Data: {data}")
        # print(f"Synth: {synth}")
        # print(f"Traj: {traj.shape}")

        k = data.shape[1]
        grid_size = int(k**0.5) + (0 if (k**0.5).is_integer() else 1)

        with PdfPages(savedir + f'/results_{suffix}.pdf') as pdf:
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
            axes = axes.flatten()

            for i in range(k):
                sns.histplot(data[:, i], kde=True, ax=axes[i], color="skyblue", edgecolor="black")
                sns.histplot(synth[:, i], kde=True, ax=axes[i], color="purple", edgecolor="red")
                axes[i].set_title(f"Column {i+1}")
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")

            for j in range(k, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

#
# from sklearn.metrics import classification_report, r2_score, f1_score
# import numpy as np
# import os
# from sklearn.utils import shuffle
# import zero
# from pathlib import Path
# import lib
# from tab_ddpm.modules import MLP
# from skorch.regressor import NeuralNetRegressor
# from skorch.classifier import NeuralNetClassifier
# from skorch.dataset import Dataset as SkDataset
# from skorch.callbacks import EarlyStopping, EpochScoring
# from skorch.helper import predefined_split
# from torch.optim import AdamW
# from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
#
# def train_mlp(
#     parent_dir,
#     real_data_path,
#     eval_type,
#     T_dict,
#     params = None,
#     change_val = False,
#     seed = 0,
#     device = "cuda:0"
# ):
#
#     synthetic_data_path = os.path.join(parent_dir) if parent_dir is not None else None
#     info = lib.load_json(os.path.join(real_data_path, 'info.json'))
#     T = lib.Transformations(**T_dict)
#
#     if change_val:
#         X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = lib.read_changed_val(real_data_path, val_size=0.2)
#
#     X = None
#     print('-'*100)
#     if eval_type == 'merged':
#         print('loading merged data...')
#         if not change_val:
#             X_num_real, X_cat_real, y_real = lib.read_pure_data(real_data_path)
#         X_num_fake, X_cat_fake, y_fake = lib.read_pure_data(synthetic_data_path)
#         y = np.concatenate([y_real, y_fake], axis=0)
#
#         X_num = None
#         if X_num_real is not None:
#             X_num = np.concatenate([X_num_real, X_num_fake], axis=0)
#
#         X_cat = None
#         if X_cat_real is not None:
#             X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)
#
#     elif eval_type == 'synthetic':
#         print('loading synthetic data...')
#         X_num, X_cat, y = lib.read_pure_data(synthetic_data_path)
#
#     elif eval_type == 'real':
#         print('loading real data...')
#         if not change_val:
#             X_num, X_cat, y = lib.read_pure_data(real_data_path)
#     else:
#         raise "Choose eval method"
#
#     if not change_val:
#         X_num_val, X_cat_val, y_val = lib.read_pure_data(real_data_path, 'val')
#     X_num_test, X_cat_test, y_test = lib.read_pure_data(real_data_path, 'test')
#
#     D = lib.Dataset(
#         {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
#         {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
#         {'train': y, 'val': y_val, 'test': y_test},
#         {},
#         lib.TaskType(info['task_type']),
#         info.get('n_classes')
#     )
#
#     D = lib.transform_dataset(D, T, None)
#     X = lib.concat_features(D)
#
#     X["train"], D.y["train"] = shuffle(X["train"], D.y["train"], random_state=seed)
#     print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')
#
#     if params is None:
#         params = lib.load_json(f"/home/rototo/tab-diffusion/tuned_models/mlp/{Path(real_data_path).name}_cv.json")
#
#     mlp_params = {}
#     if params is not None:
#         mlp_params["d_layers"] = params["d_layers"]
#         mlp_params["dropout"] = params["dropout"]
#         # mlp_params["n_blocks"] = params["n_blocks"]
#         # mlp_params["d_main"] = params["d_main"]
#         # mlp_params["d_hidden"] = params["d_hidden"]
#         # mlp_params["dropout_first"] = params["dropout_first"]
#         # mlp_params["dropout_second"] = params["dropout_second"]
#         mlp_params["d_in"] = X["train"].shape[1]
#         mlp_params["d_out"] = D.nn_output_dim
#
#     model = MLP.make_baseline(**mlp_params)
#
#     if D.is_regression:
#         y = {k: D.y[k].reshape(-1, 1).astype(np.float32) for k in D.y}
#     elif D.is_binclass:
#         y = {k: D.y[k].reshape(-1, 1).astype(np.float32) for k in D.y}
#     else:
#         y = {k: D.y[k].astype(np.int64) for k in D.y}
#
#     train_ds = SkDataset(X = X["train"].to_numpy(), y = y["train"])
#     val_ds = SkDataset(X = X["val"].to_numpy(), y = y["val"])
#     es = EarlyStopping(monitor="valid_loss", patience=16)
#
#     print('-'*100)
#
#     def f1(net, X, y):
#         y_pred = net.predict(X)
#         return f1_score(y, y_pred, average="macro")
#
#     def r2(net, X, y):
#         y_pred = net.predict(X)
#         return r2_score(y, y_pred)
#
#     if D.is_regression:
#         net = NeuralNetRegressor(
#             model,
#             criterion=MSELoss,
#             optimizer=AdamW,
#             lr=params["lr"],
#             optimizer__weight_decay=params["weight_decay"],
#             batch_size=128 if len(D.y["train"]) < 10_000 else 256,
#             max_epochs=1000,
#             train_split=predefined_split(val_ds),
#             iterator_train__shuffle=True,
#             device=device,
#             callbacks=[es, EpochScoring(r2, lower_is_better=False)],
#         )
#
#     else:
#         net = NeuralNetClassifier(
#             model,
#             criterion=BCEWithLogitsLoss if D.is_binclass else CrossEntropyLoss,
#             optimizer=AdamW,
#             lr=params["lr"],
#             optimizer__weight_decay=params["weight_decay"],
#             batch_size=128 if len(D.y["train"]) < 10_000 else 256,
#             max_epochs=1000,
#             train_split=predefined_split(val_ds),
#             iterator_train__shuffle=True,
#             device=device,
#             callbacks=[es, EpochScoring(f1, lower_is_better=False)],
#         )
#
#     net.fit(
#         X=train_ds.X,
#         y=train_ds.y
#     )
#
#     print("LAST:", len(net.history))
#
#     predictions = {k: net.predict_proba(v.to_numpy())[:, 1] if D.is_binclass else 
#                       net.predict_proba(v.to_numpy()) if D.is_multiclass else 
#                       net.predict(v.to_numpy()) 
#         for k, v in X.items()
#     }
#
#     report = {}
#     report['eval_type'] = eval_type
#     report['dataset'] = real_data_path
#     report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')
#
#     metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
#     metrics_report.print_metrics()
#
#     if parent_dir is not None:
#         lib.dump_json(report, os.path.join(parent_dir, "results_mlp.json"))
#
#     return metrics_report
