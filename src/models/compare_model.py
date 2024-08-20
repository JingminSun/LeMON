from sklearn.cross_decomposition import CCA
import numpy as np
import torch


def compute_cca_similarity(layer_outputs1, layer_outputs2, n_components=5):
    # Ensure that the inputs have at least 2 samples
    if layer_outputs1.shape[0] < 2 or layer_outputs2.shape[0] < 2:
        raise ValueError("Each input must have at least 2 samples.")

    # Reshape the inputs if necessary (assuming they might be higher dimensional)
    layer_outputs1 = layer_outputs1.reshape(layer_outputs1.shape[0], -1)
    layer_outputs2 = layer_outputs2.reshape(layer_outputs2.shape[0], -1)

    # Initialize and fit the CCA model
    cca = CCA(n_components=n_components)
    cca.fit(layer_outputs1, layer_outputs2)

    # Transform the inputs to the canonical space
    X_c, Y_c = cca.transform(layer_outputs1, layer_outputs2)

    # Compute the correlation matrix between the canonical variables
    correlation_matrix = np.corrcoef(X_c.T, Y_c.T)

    # Extract the diagonal elements that correspond to the canonical correlations
    canonical_correlations = correlation_matrix.diagonal(offset=X_c.shape[1])

    # Return the mean of the canonical correlations
    return np.mean(canonical_correlations)

def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    return (X.T @ Y) ** 2 / (norm_X ** 2 * norm_Y ** 2)

def compare_parameters_with_similarity(output1_dict, output2_dict):
    cca_dict = {}
    # cka_dict = {}
    for k in output1_dict.keys():
        output1 = output1_dict[k]
        output2 = output2_dict[k]
        if output1.dtype == torch.bfloat16:
            output1 = output1.to(torch.float32)
        if output2.dtype == torch.bfloat16:
            output2 = output2.to(torch.float32)
        bs = output1.size(0)

        output1_np = output1.detach().cpu().numpy().reshape((bs,-1))
        output2_np = output2.detach().cpu().numpy().reshape((bs,-1))

        # cka = 0
        cca = compute_cca_similarity(output1_np, output2_np)
        cca_dict[k] = cca

    return cca_dict#,cka_dict

#
# if __name__ == '__main__':
#     from models.build_model import build_model
#     import hydra
#     import torch
#     from symbol_utils.environment import SymbolicEnvironment
#     from data_utils.collate import custom_collate
#     from trainer import Trainer
#     from dataset import get_dataset
#     import torch.nn as nn
#     from torch.utils.data import DataLoader
#
#     class SimpleModel(nn.Module):
#         def __init__(self):
#             super(SimpleModel, self).__init__()
#             self.net = nn.Sequential(
#                 nn.Linear(1, 40),
#                 nn.ReLU(),
#                 nn.Linear(40, 1)
#             )
#
#         def forward(self, x):
#             return self.net(x)
#
#     @hydra.main(version_base=None, config_path="../configs", config_name="main")
#     def test(params):
#         params.multi_gpu =0
#         params.optim.max_iters = 5
#         params.dump_path = "checkpoint/jingmins/dumped/debug/plot"
#         params.base_seed = 0
#         params.n_gpu_per_node = 1
#         params.local_rank = 0
#         params.global_rank = 0
#         params.num_workers = 4
#         params.batch_size = 10
#         params.eval_size = 400
#         symbol_env = SymbolicEnvironment(params.symbol)
#         modules1 = build_model(params, params.model, params.data, symbol_env)
#         print("Listing all submodules:")
#         list_submodules(modules1["model"])
#         modules2 = build_model(params, params.model, params.data, symbol_env)
#         trainer = Trainer(modules1, params, symbol_env)
#         support_datasets: dict = get_dataset(params, symbol_env, split="eval",
#                                                   meta=False)
#         support = {
#             k: next(iter(DataLoader(
#                 v,
#                 batch_size=params.batch_size,
#                 num_workers=params.num_workers,
#                 # num_workers=1,
#                 # pin_memory=True,
#                 collate_fn=custom_collate(params.data.max_output_dimension, symbol_env,
#                                           meta=False)
#             )))
#             for k, v in support_datasets.items()
#         }
#         support_dict = trainer.prepare_data(support, train=True)
#         result = compare_parameters_with_similarity(modules1, modules2, support_dict,
#                                                     compute_cca_similarity)
#         print(result)
#     test()