import os
import numpy as np
from logging import getLogger
from collections import defaultdict
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.compare_model import compare_parameters_with_similarity
from dataset import get_dataset
from utils.misc import sync_tensor
from utils.metrics import compute_metrics
from utils.plot import plot_2d_pde, plot_1d_pde
from data_utils.collate import custom_collate
from tabulate import tabulate
from utils.misc import to_cuda
from models.transformer_wrappers import Combine_freeze_encoder
import wandb
# np.seterr(all="raise")
np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")

logger = getLogger()

metric_to_header = {
    "_l2_error": "rel l2",
"_l2_error_zero_shot": "z: rel l2",
"_l2_error_few_shot": "f: rel l2",
    "_mse": "mse",
    "_rmse": "rmse",
    "_l2_error_first_half": "rel l2 1st_half",
    "_l2_error_second_half": "rel l2 2nd_half",
    "_l2_error_first_half_zero_shot": "z: rel l2 1st_half",
    "_l2_error_second_half_zero_shot": "z: rel l2 2nd_half",
    "_l2_error_first_half_few_shot": "f: rel l2 1st_half",
    "_l2_error_second_half_few_shot": "f: rel l2 2nd_half",
}


def data_loss_fn(data_output, data_label, data_mask, weight=None):
    # copy of trainer data_loss_fn, by batchdi
    loss = F.mse_loss(data_output, data_label, reduction="none")
    pred_mask = data_mask.expand_as(loss)
    if weight is None:
        # no re-weighting, loss is just regular MSE
        loss = (loss * pred_mask).flatten(1).sum(1) / (pred_mask.flatten(1).sum(1))
    else:
        # reweight by weight
        weight = weight.expand_as(loss)
        loss = ((loss * pred_mask) * weight).flatten(1).sum(1)
    return loss.tolist()  # (bs, )


class Evaluator(object):

    def __init__(self, trainer, symbol_env):
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.symbol_env = symbol_env

        if self.params.data.eval_skip > -1:
            self.skip = self.params.data.eval_skip
        else:
            if self.params.zero_shot_only:
                base_skip = 0
            else:
                base_skip = self.params.eval_support_size * 20
            self.skip =  base_skip+self.params.train_size*2 if self.params.meta else  base_skip+self.params.train_size
        self.datasets: dict = get_dataset(self.params, self.symbol_env, split="eval", skip = self.skip)
        self.dataloaders = {
            k: DataLoader(
                v,
                batch_size=self.params.IC_per_param,
                num_workers=self.params.num_workers,
                # num_workers=1,
                # pin_memory=True,
                shuffle=False,
                collate_fn=custom_collate(self.params.data.max_output_dimension,symbol_env),
            )
            for k, v in self.datasets.items()
        }
        self.iteration = {
            k: iter(self.dataloaders[k]) for k in self.dataloaders.keys()
        }

        self.types = self.datasets.keys()

        self.validation_metrics = self.params.validation_metrics_print.split(",")

    @torch.enable_grad()
    def evaluate(self):

        params = self.params

        model = self.modules["model"]
        model.eval()

        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

        if params.log_eval_plots > 0:
            plot_folder = os.path.join(params.eval_dump_path, f"epoch_{self.trainer.epoch}_{self.params.local_rank}")
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)

        all_results = {}

        for type, loader in self.dataloaders.items():
            eval_size = 0
            num_plotted = 0
            results = defaultdict(list)
            # if self.params.model.name == "DeepONet" or  self.params.model.name == "FNO":
            #     assert params.zero_shot_only, "only zero shot for deeponet"
            for idx, samples in enumerate(loader):
                bs = len(samples["data"])
                eval_size += bs
                dict = (
                     self.trainer.prepare_data_deepo(samples,train=False) if self.params.model.name == "DeepONet" else self.trainer.prepare_data(samples,train=False)
                )

                # Initialize empty dictionaries for the 20-sample and 10-sample subsets
                dict_query = {}
                dict_support = {}

                # Iterate over each attribute in the original dictionary
                for key, value in dict.items():
                    # Randomly sample 30 distinct items from the original 50-item list
                    sampled_all = self.datasets[type].rng.sample(value, self.params.data.num_support+self.params.data.query)

                    # Split the sampled data into 20 and 10
                    dict_support[key] = sampled_all[:self.params.data.num_support]
                    dict_query[key] = sampled_all[self.params.data.num_support:]
                data_label = dict_query["data_label"]
                if self.params.model.name == "DeepONet":
                    with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                        output = model(
                            querypoint=dict_query["query_tensor_reshaped"],
                            value_at_sensor=dict_query["data_input_reshaped"],
                        )  # (bs, output_len, x_num, data_dim)
                        output_start = self.params.data.input_len if self.params.data.output_start_eval is None else self.params.data.output_start_eval
                        num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                        data_output_zero_shot = output.reshape(bs, num_output_t, params.data.x_num, 1)
                elif self.params.model.name == "FNO":
                    data_output_zero_shot = model(
                        dict_query["data_input"],
                    )  # (bs, output_len, x_num, data_dim
                else:
                    with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                        output_dict_zero_shot = model(
                            "generate",
                            data_input=dict_query["data_input"],
                            input_times=dict_query["input_times"][..., None],
                            output_times=dict_query["output_times"][..., None],
                            symbol_input=dict_query["symbol_input"],
                            symbol_padding_mask=dict_query["symbol_mask"]
                        )  # (bs, output_len, x_num, x_num, data_dim)
                        data_output_zero_shot = output_dict_zero_shot["data_output"]
                        data_output_zero_shot = data_output_zero_shot * dict_query["data_mask"]
                data_loss_zero_shot = data_loss_fn(data_output_zero_shot, data_label,
                                                   dict_query["data_mask"], dict_query["loss_weight"])
                results["data_loss_zero_shot"].extend(data_loss_zero_shot)

                if not params.zero_shot_only:
                    if self.params.model.name == "DeepONet":
                        learner = self.deeponet_adapt(model,dict_support)
                        with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                            output = learner(
                                querypoint=dict["query_tensor_reshaped"],
                                value_at_sensor=dict["data_input_reshaped"],
                            )  # (bs, output_len, x_num, data_dim)
                            output_start = self.params.data.input_len if self.params.data.output_start_eval is None else self.params.data.output_start_eval
                            num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                            data_output_few_shot = output.reshape(bs, num_output_t, params.data.x_num, 1)

                    elif self.params.model.name == "FNO":

                        learner = self.fno_adapt(model,dict_support)
                        data_output_few_shot = learner(
                            dict["data_input"],
                        )  # (bs, output_len, x_num, data_dim
                    else:

                        learner = self.full_adapt(model,dict_support)
                        with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                            output_dict_few_shot = learner(
                                "generate",
                                data_input=dict["data_input"],
                                input_times=dict["input_times"][..., None],
                                output_times=dict["output_times"][..., None],
                                symbol_input=dict["symbol_input"],
                                symbol_padding_mask=dict["symbol_mask"]
                            )  # (bs, output_len, x_num, x_num, data_dim)

                        data_output_few_shot = output_dict_few_shot["data_output"]
                        data_output_few_shot = data_output_few_shot * dict["data_mask"]

                    data_loss_few_shot = data_loss_fn(data_output_few_shot, data_label, dict["data_mask"], dict["loss_weight"])

                    results["data_loss_few_shot"].extend(data_loss_few_shot)
                else:
                    data_output_few_shot = None

                if self.params.normalize:
                    # denormalize data
                    eps = 1e-6
                    if not params.zero_shot_only:
                        data_output_few_shot = data_output_few_shot * (dict["std"]+ eps) + dict["mean"]
                    else:
                        data_output_few_shot = None
                    data_output_zero_shot = data_output_zero_shot * (dict["std"] + eps) + dict["mean"]
                    data_label = data_label * (dict["std"] + eps) + dict["mean"]
                if not params.zero_shot_only:
                    cur_result_few_shot = compute_metrics(
                        data_output_few_shot, data_label, metrics=params.validation_metrics_print, batched=True
                    )

                    for k in cur_result_few_shot.keys():
                        keys = k + "_few_shot"
                        if k == "_r2":
                            results[keys].append(cur_result_few_shot[k])
                        else:
                            results[keys].extend(cur_result_few_shot[k])
                else:
                    cur_result_few_shot =None

                cur_result_zero_shot = compute_metrics(
                    data_output_zero_shot, data_label, metrics=params.validation_metrics_print, batched=True
                )

                for k in cur_result_zero_shot.keys():
                    keys = k + "_zero_shot"
                    if k == "_r2":
                        results[keys].append(cur_result_zero_shot[k])
                    else:
                        results[keys].extend(cur_result_zero_shot[k])
                if params.print_outputs:
                    # plot all outputs
                    if not params.zero_shot_only:
                        data_output_few_shot = data_output_few_shot.float().numpy(
                            force=True)  # (bs, output_len//output_step, x_num,data_dim)
                    else:
                        data_output_few_shot = None

                    data_output_zero_shot = data_output_zero_shot.float().numpy(
                        force=True)  # (bs, output_len//output_step, x_num,data_dim)
                    data_all = samples["data"].numpy(
                        force=True)  # (bs, input_len + output_len, x_num,  data_dim)
                    for i in range(bs):
                        index = idx * params.batch_size_eval + i
                        if not params.zero_shot_only:
                            plot_title = "Type {} | Idx {} | zero {:.4f} | few {:.4f}".format(type, index,
                                                                        cur_result_zero_shot["_l2_error"][i],  cur_result_few_shot["_l2_error"][i])
                        else:
                            plot_title = "Type {} | Idx {} | zero {:.4f}".format(type,index,cur_result_zero_shot["_l2_error"][i])


                        plot_1d_pde(
                            data_loss_zero_shot[i],
                            data_loss_few_shot[i] if data_loss_few_shot is not None else None,
                            samples["t"][i],
                            samples["x"][i],
                            data_all[i],
                            params.data.input_len,
                            plot_title,
                            filename=f"{type}_plot_{index}",
                            folder=save_folder,
                            dim=(params.data[type.split("%")[0]]).dim,
                            input_step = params.data.input_step,
                            output_step = params.data.output_step,
                            output_start =  params.data.input_len if params.data.output_start_eval is None else params.data.output_start_eval
                        )

                if params.log_eval_plots > 0 and num_plotted < params.log_eval_plots:
                    # only plot the first element
                    if (isinstance(data_output_few_shot, np.ndarray)  or (data_output_few_shot is None and params.zero_shot_only) )and isinstance(data_output_zero_shot, np.ndarray):
                        # already converted to numpy
                        if not params.zero_shot_only:
                            output_few_shot = data_output_few_shot[0]
                        else:
                            output_few_shot = None
                        output_zero_shot = data_output_zero_shot[0]
                        cur_data = data_all[0]
                    else:
                        if not params.zero_shot_only:
                            output_few_shot = data_output_few_shot[0].float().numpy(force=True)
                        else:
                            output_few_shot = None
                        output_zero_shot= data_output_zero_shot[0].float().numpy(force=True)
                        cur_data = samples["data"][0].numpy(force=True)

                    index = idx * params.batch_size_eval
                    if not params.zero_shot_only:
                        plot_title = "Type {} | Idx {} | zero {:.4f} | few {:.4f}".format(type, index,
                                                                                      cur_result_zero_shot[ "_l2_error"][0],cur_result_few_shot[ "_l2_error"][ 0])
                    else:
                        plot_title = "Type {} | Idx {} | zero {:.4f} ".format(type,index,cur_result_zero_shot[ "_l2_error"][ 0])
                    path = plot_1d_pde(
                        output_zero_shot,
                        output_few_shot if output_few_shot is not None else None,
                        samples["t"][0],
                        samples["x"][0],
                        cur_data,
                        params.data.input_len,
                        plot_title,
                        filename=f"{type}_plot_{index}",
                        folder=plot_folder,
                        dim=(params.data[type.split("%")[0]]).dim,
                        input_step=params.data.input_step,
                        output_step=params.data.output_step,
                        output_start= params.data.input_len if params.data.output_start_eval is None else params.data.output_start_eval
                    )

                    if params.use_wandb:
                        wandb.log(
                            {"val": {"epoch": self.trainer.epoch,
                                     f"{type}_plot_{num_plotted}": wandb.Image(path)}}
                        )

                    num_plotted += 1

                if params.eval_size_get > 0 and eval_size >= params.eval_size_get:
                    break

            for k, v in results.items():
                if k.startswith("_r2"):
                    results[k] = np.mean(np.array(v))
                results[k] = np.sum(np.array(v))
            results["size"] = eval_size
            all_results[type] = results

        if params.multi_gpu:
            # sync results on all gpus
            sorted_keys = None
            for type, results in all_results.items():

                if sorted_keys is None:
                    sorted_keys = sorted(results.keys())

                stats = torch.Tensor([results[k] for k in sorted_keys])
                stats = sync_tensor(stats)
                results = {k: stats[i].item() for i, k in enumerate(sorted_keys)}
                results["size"] = int(results["size"])
                all_results[type] = results

        # aggregate results and compute averages

        total_size = 0
        results_per_type = {}
        stats = defaultdict(float)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                elif k.startswith("_r2"):
                    res_mean_type[k] = v
                    stats[k] += v / len(self.dataloaders)
                elif k == "_mse":
                    # rescale mse due to padding dimensions
                    ratio = self.params.data.max_output_dimension / (self.params.data[type.split("%")[0]]).dim
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                elif k == "_rmse":
                    ratio = (self.params.data.max_output_dimension / (self.params.data[type.split("%")[0]]).dim) ** 0.5
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k] += v
            results_per_type[type] = res_mean_type
        stats = {k: v if k.startswith("_r2") else v / total_size for k, v in stats.items()}

        # report metrics per equation type as a table
        if not params.zero_shot_only:
            headers = ["type", "dim", "size", "data_loss_zero_shot", "data_loss_few_shot"] + [k+"_zero_shot" for k in self.validation_metrics] + [k+"_few_shot" for k in self.validation_metrics]
        else:
            headers = ["type", "dim", "size", "data_loss_zero_shot"] + [k + "_zero_shot" for k in self.validation_metrics]
        # if self.params.model.name == "DeepONet":
        #     headers = headers + ["direct_l2"]
        table = []
        for type, results in results_per_type.items():
            if params.data.mixed_task:
                dim_list =[(params.data[t]).dim for t in type.split('+')]
                dim = max(dim_list)
            else:
                dim = (params.data[type.split("%")[0]]).dim
            row = [type,dim]
            for k in headers[2:]:
                row.append(results[k])
            table.append(row)

        headers = list(map(lambda s: metric_to_header[s] if s in metric_to_header else s, headers))
        logger.info(
            "Evaluation Stats (total size = {})\n{}".format(
                total_size, tabulate(table, headers=headers, tablefmt="grid")
            )
        )

        return stats, results_per_type


    def full_adapt(self,model, support_dict):
        params = self.params
        num_inner_loops = params.num_adapt_eval


        learner = model.clone()
        learner = to_cuda(learner)
        # learner = copy.deepcopy(model)
        # learner.eval()
        learner.train()
        # inner_optimizer = torch.optim.SGD(learner.parameters(),
        #                                   lr=params.model.meta.meta_lr)
        for _ in range(params.meta_step):
            with torch.cuda.amp.autocast(enabled=bool(params.amp),
                                         dtype=torch.bfloat16):
                output_dict = learner(
                    "fwd",
                    data_input=support_dict["data_input"],
                    input_times=support_dict["input_times"][..., None],
                    output_times=support_dict["output_times"][..., None],
                    symbol_input=support_dict["symbol_input"],
                    symbol_padding_mask=support_dict["symbol_mask"]
                )
                data_output = output_dict["data_output"]
                support_data_loss = self.trainer.data_loss_fn(data_output,
                                                              support_dict[
                                                                  "data_label"],
                                                              support_dict[
                                                                  "data_mask"],
                                                              support_dict[
                                                                  "loss_weight"])
                learner.adapt(support_data_loss, lr=self.params.model.meta.meta_lr)

        return learner


    def deeponet_adapt(self,model,dict_support):
        params = self.params
        num_inner_loops = params.num_adapt_eval

        learner = model.clone()
        learner = to_cuda(learner)
        learner.train()
        for _ in range(params.meta_step):
            with torch.cuda.amp.autocast(enabled=bool(params.amp),
                                         dtype=torch.bfloat16):
                output = learner(
                    querypoint=dict_support["query_tensor_reshaped"],
                    value_at_sensor=dict_support["data_input_reshaped"],
                )
                output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
                num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                data_output = output.reshape(params.eval_support_size, num_output_t, params.data.x_num, 1)
                support_data_loss = self.trainer.data_loss_fn(data_output, dict_support["data_label"],
                                                      dict_support["data_mask"], dict_support["loss_weight"])

            learner.adapt(support_data_loss, lr=self.params.model.meta.meta_lr)
        return learner
    def fno_adapt(self,model,dict_support):
        params = self.params
        param_accumulator = None
        num_inner_loops = params.num_adapt_eval

        learner = model.clone()
        learner = to_cuda(learner)
        learner.train()
        for _ in range(params.meta_step):

            output = learner(
                dict_support["data_input"],
            )  # (bs, output_len, x_num, data_dim)
            output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
            num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
            data_output = output.reshape(params.eval_support_size, num_output_t, params.data.x_num, 1)
            support_data_loss = self.trainer.data_loss_fn(data_output, dict_support["data_label"],
                                                  dict_support["data_mask"], dict_support["loss_weight"])

            learner.adapt(support_data_loss, lr=self.params.model.meta.meta_lr)
        return learner
    def freeze_adapt_encoder(self,model,type):


        params = self.params
        param_accumulator = None
        num_inner_loops = params.num_adapt_eval

        for __ in range(num_inner_loops):
            learner = model.inner_model.clone()
            learner = to_cuda(learner)
            learner.eval()
            if params.data.diff_supp_query_type:
                type_idx = params.data.eval_types_query.index(type)
                type_support = params.data.eval_types_support[type_idx]
                support_data = self.support_iter[type_support]
            else:
                try:
                    support_data = next(self.support_iter[type])
                except:
                    # logger.info(f"Reached end of dataloader, restart {self.dataloader_count}...")
                    self.support_iter[type] =  iter(DataLoader(
                        self.support_datasets[type],
                        batch_size=self.params.eval_support_size,
                        num_workers=self.params.num_workers,
                        # num_workers=1,
                        # pin_memory=True,
                        shuffle=True,
                        collate_fn=custom_collate(self.params.data.max_output_dimension, self.symbol_env)
                    ))
                    support_data = next(self.support_iter[type])
            support_dict = self.trainer.prepare_data(support_data, train=True)
            output_noinner = model.no_inner_model(
                        data_input= support_dict["data_input"],
                        input_times= support_dict["input_times"][..., None],
                        symbol_input=support_dict["symbol_input"],
                        symbol_padding_mask=support_dict["symbol_mask"],)
            with torch.cuda.amp.autocast(enabled=bool(params.amp),
                                         dtype=torch.bfloat16):
                adapt_before = learner(
                    "fwd",
                    output_times=support_dict["output_times"][..., None],
                    embedder=model.no_inner_model.embedder,
                    fused=output_noinner["fused"],
                    fused_mask=output_noinner["fused_mask"]
                )
            learner.train()
            for _ in range(params.meta_step):
                with torch.cuda.amp.autocast(enabled=bool(params.amp),
                                             dtype=torch.bfloat16):
                    output_dict = learner(
                        "fwd",
                        output_times=support_dict["output_times"][..., None],
                        embedder=model.no_inner_model.embedder,
                        fused=output_noinner["fused"],
                        fused_mask=output_noinner["fused_mask"]
                    )
                    data_output = output_dict["data_output"]
                    support_data_loss = self.trainer.data_loss_fn(data_output,
                                                                  support_dict[
                                                                      "data_label"],
                                                                  support_dict[
                                                                      "data_mask"],
                                                                  support_dict[
                                                                      "loss_weight"])
                    learner.adapt(support_data_loss, lr=self.params.model.meta.meta_lr)
            learner.eval()
            adapt_after = learner(
                "fwd",
                output_times=support_dict["output_times"][..., None],
                embedder=model.no_inner_model.embedder,
                fused=output_noinner["fused"],
                fused_mask=output_noinner["fused_mask"]
            )
            cca_dict = compare_parameters_with_similarity(adapt_before, adapt_after)
            logger.info("cca similarity")
            for key, value in cca_dict.items():
                logger.info(f"{key}: {value}")

            if param_accumulator is None:
                param_accumulator = {name: torch.zeros_like(param) for name, param in
                                     learner.named_parameters()}

            for name, param in learner.named_parameters():
                param_accumulator[name] += param.detach()

        # Calculate mean of accumulated parameters
        for name in param_accumulator:
            param_accumulator[name] /= num_inner_loops

        averaged_model = model.clone()
        # Update model with mean parameters
        for name, param in averaged_model.named_parameters():
            param.data = param_accumulator[name]
        return averaged_model

    # def freeze_adapt_symbol(self,model,type):
    #
    #
    #     params = self.params
    #     learner = model.inner_model.clone()
    #     learner = to_cuda(learner)
    #     learner.eval()
    #     support_data = self.support[type]
    #     support_dict = self.trainer.prepare_data(support_data, train=True)
    #     output_noinner = model.no_inner_model("fwd",
    #                                           symbol_input=support_dict["symbol_input"],
    #                                           symbol_padding_mask=support_dict["symbol_mask"])
    #     with torch.cuda.amp.autocast(enabled=bool(params.amp),
    #                                  dtype=torch.bfloat16):
    #         adapt_before = learner(
    #             "fwd",
    #             data_input=support_dict["data_input"],
    #             input_times=support_dict["input_times"][..., None],
    #             output_times=support_dict["output_times"][..., None],
    #             symbol_encoded=output_noinner["symbol_encoded"],
    #             symbol_padding_mask=support_dict["symbol_mask"]
    #         )
    #     learner.train()
    #     # inner_optimizer = torch.optim.SGD(learner.parameters(),
    #     #                                   lr=params.model.meta.meta_lr)
    #     for _ in range(params.meta_step):
    #         with torch.cuda.amp.autocast(enabled=bool(params.amp),
    #                                      dtype=torch.bfloat16):
    #             output_dict = learner(
    #             "fwd",
    #             data_input=support_dict["data_input"],
    #             input_times=support_dict["input_times"][..., None],
    #             output_times=support_dict["output_times"][..., None],
    #             symbol_encoded=output_noinner["symbol_encoded"],
    #             symbol_padding_mask=support_dict["symbol_mask"]
    #             )
    #             data_output = output_dict["data_output"]
    #             support_data_loss = self.trainer.data_loss_fn(data_output,
    #                                                           support_dict[
    #                                                               "data_label"],
    #                                                           support_dict[
    #                                                               "data_mask"],
    #                                                           support_dict[
    #                                                               "loss_weight"])
    #             learner.adapt(support_data_loss,lr = self.params.model.meta.meta_lr_eval)
    #             # learner.zero_grad()
    #             # support_data_loss.backward(retain_graph= not self.params.model.meta.first_order)
    #             # inner_optimizer.step()
    #             # self.trainer.optimize(support_data_loss, optimizer=inner_optimizer,
    #             #                       outer_loop=False)
    #     learner.eval()
    #     adapt_after = learner(
    #             "fwd",
    #             data_input=support_dict["data_input"],
    #             input_times=support_dict["input_times"][..., None],
    #             output_times=support_dict["output_times"][..., None],
    #             symbol_encoded=output_noinner["symbol_encoded"],
    #             symbol_padding_mask=support_dict["symbol_mask"]
    #         )
    #     cca_dict = compare_parameters_with_similarity(adapt_before, adapt_after)
    #     logger.info("cca similarity")
    #     for key, value in cca_dict.items():
    #         logger.info(f"{key}: {value}")
    #
    #     return learner