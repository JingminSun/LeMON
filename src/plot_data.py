import sys
import os
from models.build_model import build_model
from symbol_utils.environment import SymbolicEnvironment
from dataset import get_dataset
from omegaconf import DictConfig, OmegaConf
from utils.metrics import compute_metrics
from utils.mode import init_distributed_mode
from matplotlib import pyplot as plt
import hydra
import torch
from pathlib import Path
import  numpy as np
from utils.misc import to_cuda, initialize_exp
from torch.utils.data import DataLoader
from data_utils.collate import custom_collate

def prepare_data_deepo( params,samples, train=True):
    data = samples["data"]
    data_mask = samples["data_mask"][:, None, None, :]  # (bs, 1, 1, dim)
    query_locations = samples["query"]

    input_len = params.data.input_len
    input_step = params.data.input_step
    output_step = params.data.output_step
    if train:
        output_start = input_len if params.data.output_start is None else params.data.output_start
    else:
        output_start = input_len if  params.data.output_start_eval is None else  params.data.output_start_eval
    data_label = data[:, output_start::output_step, :]

    # Example of data_input and sensors
    data_input = data[:, :input_len:input_step, :]

    data_input, data_label, data_mask, query_locations = to_cuda(
        (data_input, data_label, data_mask, query_locations)
    )

    if  params.normalize:
        mean = torch.mean(data_input, axis=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
        std = torch.std(data_input, axis=(1, 2), keepdim=True)  # (bs, 1, 1, dim)

        eps = 1e-6
        data_input = (data_input - mean) / (std + eps)
        data_label = (data_label - mean) / (std + eps)  # use same mean and std

    else:
        mean = 0
        std = 1

    # prepare weights for loss function
    eps = 1e-5
    if params.loss_weight is None or  params.loss_weight == "none":
        loss_weight = None
    elif  params.loss_weight == "l2":
        # reweight by L2 norm squared
        loss_weight = torch.linalg.vector_norm(data_label, dim=(1, 2), keepdim=True) ** 2  # (bs, 1, 1, dim)
    elif  params.loss_weight == "linfty":
        # reweight by L-infinity norm
        loss_weight, _ = torch.max(torch.abs(data_label), dim=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
    else:
        assert False, f"Unknown reweight type: {params.loss_weight}"

    if loss_weight is not None:
        bs = np.single(data_label.size(0))
        # loss_weight = to_cuda(
        #     (torch.reciprocal(loss_weight + eps) / bs).expand_as(data_label).float()
        # )  # (bs, output_len, x_num, x_num, dim)
        loss_weight = to_cuda((torch.reciprocal(loss_weight + eps) / bs).float())  # (bs, 1,  1, dim)

    data_input_reshaped = data_input.repeat_interleave(query_locations.shape[1], dim=0)
    query_tensor_reshaped = query_locations.view(-1, 1, 2)
    result_tensor_reshaped = data_label.view(-1, 1, 1)
    dict = {
        "data_input_reshaped": data_input_reshaped,
        "query_tensor_reshaped": query_tensor_reshaped,
        "data_label_reshaped": result_tensor_reshaped,
        "data_input": data_input,
        "data_label": data_label,
        "mean": mean,
        "std": std,
        "query_point": query_locations,
        "loss_weight": loss_weight,
        "data_mask": data_mask
    }

    return dict


def prepare_data(params,samples, train=True):
    """
    Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)

    samples: data:         Tensor     (bs, max_len, x_num,  dim)
             data_mask:    BoolTensor (bs, dim)
             t:            Tensor     (bs, max_len)

    """

    data = samples["data"]
    data_mask = samples["data_mask"][:, None, None, :]  # (bs, 1, 1, dim)
    t = samples["t"]
    symbol = samples["tree_encoded"]

    symbol_input = symbol[:, 1:-1]  # Deleting EOS/BOS
    symbol_mask = samples["tree_mask"][:, 1:-1]

    input_len = params.data.input_len
    input_step = params.data.input_step
    output_step =params.data.output_step
    if train:
        output_start = input_len if params.data.output_start is None else params.data.output_start
    else:
        output_start = input_len if params.data.output_start_eval is None else params.data.output_start_eval

    data_input = data[:, :input_len:input_step]  # (bs, input_len, x_num,  dim)
    data_label = data[:, output_start::output_step]  # (bs, output_len, x_num,  dim)
    input_times = t[:, :input_len:input_step]  # (bs, input_len)
    output_times = t[:, output_start::output_step]  # (bs, output_len)

    data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask = to_cuda(
        (data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask)
    )

    if params.normalize:
        mean = torch.mean(data_input, axis=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
        std = torch.std(data_input, axis=(1, 2), keepdim=True)  # (bs, 1, 1, dim)

        eps = 1e-6
        data_input = (data_input - mean) / (std + eps)
        data_label = (data_label - mean) / (std + eps)  # use same mean and std

    else:
        mean = 0
        std = 1

    # prepare weights for loss function
    eps = 1e-5
    if params.loss_weight is None or params.loss_weight == "none":
        loss_weight = None
    elif params.loss_weight == "l2":
        # reweight by L2 norm squared
        loss_weight = torch.linalg.vector_norm(data_label, dim=(1, 2), keepdim=True) ** 2  # (bs, 1, 1, dim)
    elif params.loss_weight == "linfty":
        # reweight by L-infinity norm
        loss_weight, _ = torch.max(torch.abs(data_label), dim=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
    else:
        assert False, f"Unknown reweight type: {params.loss_weight}"

    if loss_weight is not None:
        bs = np.single(data_label.size(0))
        # loss_weight = to_cuda(
        #     (torch.reciprocal(loss_weight + eps) / bs).expand_as(data_label).float()
        # )  # (bs, output_len, x_num, x_num, dim)
        loss_weight = to_cuda((torch.reciprocal(loss_weight + eps) / bs).float())  # (bs, 1,  1, dim)

    dict = {
        "data_input": data_input,
        "data_label": data_label,
        "mean": mean,
        "std": std,
        "input_times": input_times,
        "output_times": output_times,
        "data_mask": data_mask,
        "loss_weight": loss_weight,
        "symbol_input": symbol_input,
        "symbol_mask": symbol_mask
    }

    return dict


def plot_1d_pde(
    time,
    coords,
    data_all,
    input_len,
    plot_title,
    filename,
    folder="",
    dim=-1,
    input_step=1,
    output_step=1,
    output_start=None,
):
    """
    Plot 1D PDE data including input, target, output, and difference.
    If output_2 is None, only plots related to output_1 are generated.
    - output: (output_len//output_step, x_num, data_dim)
    - data_all: (input_len + output_len, x_num, data_dim)
    - time: (input_len + output_len)
    """
    if output_start is None:
        output_start = input_len
    if dim < 0:
        dim = data_all.shape[-1]

    # Ensure time and coords are numpy arrays
    time = np.array(time)
    coords = np.array(coords)

    # Slice data for input and target
    input = data_all[:input_len:input_step]
    all = data_all[::output_step]
    all[all.shape[0]//2:] = 0
    input_time = time[:input_len:input_step]
    time = time[::output_step]

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # ax.invert_yaxis()

    # Assuming `dim` is 1 since you only have one plot
    data = all[..., 0]
    # Create the plot
    im = ax.imshow(data, aspect='auto', origin='lower')

    # Set up tick positions and labels
    # num_x_ticks = 10
    # num_y_ticks = 5
    # x_tick_positions = np.linspace(0, data.shape[1] - 1, num=num_x_ticks, dtype=int)
    # y_tick_positions = np.linspace(0, data.shape[0] - 1, num=num_y_ticks, dtype=int)
    # x_tick_labels = [f"{coords[idx]:.2f}" for idx in x_tick_positions]
    # y_tick_labels = [f"{time[idx]:.2f}" for idx in y_tick_positions]
    #
    # ax.set_xticks(x_tick_positions)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_yticklabels(y_tick_labels)

    # Add colorbar
    # plt.colorbar(im, ax=ax)

    # Set the title and layout
    plt.suptitle(plot_title, fontsize=20)
    plt.tight_layout()

    # Save the figure
    path = os.path.join(folder, filename + ".png")
    plt.savefig(path)
    plt.close(fig)

    return path
@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(params: DictConfig):
    params.zero_shot_only=1
    params.meta=0
    symbol_env = SymbolicEnvironment(params)

    init_distributed_mode(params)

    initialize_exp(params)


    skip = params.eval_support_size * 20+params.train_size
    dataset = get_dataset(params, symbol_env, split = "eval", skip = skip,meta = params.meta)

    dataloaders = {
        k: DataLoader(
            v,
            batch_size=params.batch_size_eval,
            num_workers=params.num_workers,
            # num_workers=1,
            # pin_memory=True,
            # shuffle=True,
            collate_fn=custom_collate(params.data.max_output_dimension, symbol_env),
        )
        for k, v in dataset.items()
    }

    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)

    plot_folder = params.eval_dump_path


    for type, loader in dataloaders.items():
        eval_size = 0
        num_plotted = 0

        with torch.no_grad():
            for idx, samples in enumerate(loader):
                bs = len(samples["data"])
                eval_size += bs
                dict_prose= prepare_data(params, samples, train=False)
                data_label = dict_prose["data_label"]



                index = idx * params.batch_size_eval
                path = plot_1d_pde(
                    samples["t"][0],
                    samples["x"][0],
                    samples["data"].cpu().numpy(force=True)[0],
                    params.data.input_len,
                    plot_title="",
                    filename=f"{type}_plot_{index}",
                    folder=plot_folder,
                    dim=(params.data[type.split("%")[0]]).dim,
                    input_step=params.data.input_step,
                    output_step=params.data.output_step,
                    output_start=params.data.input_len if params.data.output_start_eval is None else params.data.output_start_eval
                )


if __name__ == '__main__':
    main()