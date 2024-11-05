import os
from logging import getLogger
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.optim import get_optimizer
from transformers import get_scheduler
from utils.misc import to_cuda
from data_utils.collate import custom_collate
from dataset import get_dataset
# from torch.backends.cuda import sdp_kernel
import copy
import concurrent.futures
logger = getLogger()


class Trainer(object):
    def __init__(self, modules, params, symbol_env):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.params = params
        self.symbol_env = symbol_env

        # epoch / iteration size
        self.n_steps_per_epoch = params.n_steps_per_epoch
        self.inner_epoch = 0

        # set parameters
        self.set_parameters()

        # distributed

        if params.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                    # find_unused_parameters=True,
                )

        # set optimizer
        self.set_optimizer()

        # amp
        self.scaler = None
        if params.amp:
            assert not params.cpu
            self.scaler = torch.cuda.amp.GradScaler()

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m, False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-np.infty if biggest else np.infty) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0

        # reload potential checkpoints
        self.reload_checkpoint()
        self.create_dataloader()
        self.data_loss = 0.0

    def create_dataloader(self):
        params = self.params
            # create data loaders
        if not params.eval_only:
            self.dataloader_count = 0
            if not params.meta or params.data.mixed_task:
                self.dataset = get_dataset(params, self.symbol_env, split="train", meta=params.meta, mixed_type= params.data.mixed_task)
                batch_size = params.batch_size
                self.dataloader = DataLoader(
                    self.dataset,
                    batch_size=batch_size if not params.data.mixed_task else self.params.data.num_support + self.params.data.num_query ,
                    shuffle=True,
                    num_workers=params.num_workers,
                    drop_last=True,
                    # pin_memory=True,
                    collate_fn=custom_collate(params.data.max_output_dimension,self.symbol_env),
                )
                self.data_iter = iter(self.dataloader)
            else:
                self.datasets_support: dict = get_dataset(self.params, self.symbol_env, split="train", meta=params.meta, support=1)
                self.datasets_query: dict = get_dataset(self.params, self.symbol_env, split="train", meta=params.meta,
                                                          support=0,skip=self.params.train_size)
                self.dataloaders_support = {
                    k: DataLoader(
                        v,
                        batch_size=self.params.data.num_support,
                        num_workers=self.params.num_workers,
                        # num_workers=1,
                        # pin_memory=True,
                        collate_fn=custom_collate(self.params.data.max_output_dimension, self.symbol_env),
                        shuffle=True
                    )
                    for k, v in self.datasets_support.items()
                }
                self.dataloaders_query= {
                    k: DataLoader(
                        v,
                        batch_size=self.params.data.num_query,
                        num_workers=self.params.num_workers,
                        # num_workers=1,
                        # pin_memory=True,
                        collate_fn=custom_collate(self.params.data.max_output_dimension, self.symbol_env),
                        shuffle=True
                    )
                    for k, v in self.datasets_query.items()
                }
                self.data_iters_support = {
                    k: iter(self.dataloaders_support[k]) for k in self.dataloaders_support.keys()
                }

                self.data_iters_query = {
                    k: iter(self.dataloaders_query[k]) for k in self.dataloaders_query.keys()
                }


    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            num = sum([torch.numel(p) for p in v])
            logger.info(f"Found {num:,} parameters in {k}.")
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.scheduler = None
        if params.optim.type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters["model"],
                lr=params.optim.lr,
                weight_decay=params.optim.weight_decay,
                eps=params.optim.get("eps", 1e-8),
                amsgrad=params.optim.get("amsgrad", False),
            )
            if params.optim.scheduler_type:
                self.scheduler = get_scheduler(
                    name=params.optim.scheduler_type,
                    optimizer=self.optimizer,
                    num_warmup_steps=params.optim.warmup,
                    num_training_steps=params.optim.max_iters,
                )
            logger.info(f"Optimizer: {type(self.optimizer)}, scheduler: {type(self.scheduler)}")
        else:
            self.optimizer = get_optimizer(self.parameters["model"], params.optim.lr, params.optim.type)
            logger.info(f"Optimizer: {type(self.optimizer)}")

    def optimize(self, loss):#,learner=None,optimizer=None, outer_loop = True):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            exit()

        params = self.params

        # optimizer
        # if optimizer is None:
        optimizer = self.optimizer
        # else:
        #     optimizer = optimizer



        if params.accumulate_gradients > 1:
            loss = loss / params.accumulate_gradients

        # regular optimization
        if not params.amp:
            loss.backward(retain_graph=not self.params.model.meta.first_order)
            # if self.params.meta and outer_loop:
            #     assert learner is not None
            #     for model_param, learner_param in zip(self.parameters["model"], learner.parameters()):
            #         if model_param.grad is None:
            #             model_param.grad = learner_param.grad.clone()
            #         else:
            #             model_param.grad += learner_param.grad.clone()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                optimizer.step()
                if self.scheduler is not None:# and outer_loop:
                    self.scheduler.step()
                optimizer.zero_grad()

        # AMP optimization
        else:
            self.scaler.scale(loss).backward(retain_graph=not self.params.model.meta.first_order)
            # if self.params.meta and outer_loop:
            #     assert learner is not None
            #     for model_param, learner_param in zip(self.parameters["model"],
            #                                           learner.parameters()):
            #         if model_param.grad is None:
            #             model_param.grad = learner_param.grad.clone()
            #         else:
            #             model_param.grad += learner_param.grad.clone()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                if self.scheduler is not None:# and outer_loop:
                    self.scheduler.step()
                optimizer.zero_grad()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.params.print_freq != 0:
            return

        s_iter = "%7i - " % self.n_total_iter

        # learning rates
        s_lr = (" - LR: ") + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)

        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)

        logger.info(s_iter + s_mem + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, f"{name}.pth")
        logger.info(f"Saving {name} to {path} ...")

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "dataloader_count": self.dataloader_count,
            "best_metrics": self.best_metrics,
            "params": {k: v for k, v in self.params.__dict__.items()},
            "model": self.modules["model"].state_dict()
        }
        # if self.params.meta and hasattr(self, 'meta_learner'):
        #     data["meta_learner"] = self.meta_learner.state_dict()
        for k, v in self.modules.items():
            # logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            logger.warning(f"Saving model and optimizer parameters ...")
        else:
            logger.warning(f"Saving model parameters ...")

        torch.save(data, path)

    def reload_checkpoint(self, path=None, root=None, requires_grad=True):
        """
        Reload a checkpoint if we find one.
        """
        if path is None:
            path = "checkpoint.pth"

        if self.params.reload_checkpoint is not None:
            checkpoint_path = os.path.join(self.params.reload_checkpoint, path)
            assert os.path.isfile(checkpoint_path)
        else:
            if root is not None:
                checkpoint_path = os.path.join(root, path)
            else:
                checkpoint_path = os.path.join(self.params.dump_path, path)
            if not os.path.isfile(checkpoint_path):
                logger.warning("Checkpoint path does not exist, {}".format(checkpoint_path))
                return

        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            try:
                weights = data[k]
                v.load_state_dict(weights)
            except RuntimeError:  # remove the 'module.'
                weights = {name.partition(".")[2]: v for name, v in data[k].items()}
                v.load_state_dict(weights)
            v.requires_grad = requires_grad


        # reload optimizer
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        if "scaler" in data and self.scaler is not None:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])

        if "scheduler" in data and self.scheduler is not None:
            logger.warning("Reloading scheduler...")
            self.scheduler.load_state_dict(data["scheduler"])
        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.dataloader_count = data["dataloader_count"]
        self.best_metrics = data["best_metrics"]
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and  (self.epoch +1 ) % self.params.save_periodic == 0:
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores, prefix=None, suffix=None):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            _metric = metric + "_zero_shot"
            if prefix is not None:
                _metric = prefix + "_" + _metric
            if suffix is not None:
                _metric = _metric + "_" + suffix
            if _metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % _metric)
                continue
            factor = 1 if biggest else -1

            if metric in self.best_metrics:
                best_so_far = factor * self.best_metrics[metric]
            else:
                best_so_far = -np.inf
            if factor * scores[_metric] > best_so_far:
                self.best_metrics[metric] = scores[_metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[_metric]))
                self.save_checkpoint("best-%s" % metric)

    def end_epoch(self):
        """
        End the epoch.
        """

        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        """
        Return a training batch
        """
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.dataloader_count += 1
            # logger.info(f"Reached end of dataloader, restart {self.dataloader_count}...")
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        return batch
    def get_task(self):
        if self.params.data.mixed_task:
            try:
                batch = next(self.data_iter)
            except StopIteration:
                self.dataloader_count += 1
                # logger.info(f"Reached end of dataloader, restart {self.dataloader_count}...")
                self.data_iter = iter(self.dataloader)
                batch = next(self.data_iter)

            batch_support = {
                key: value[:self.params.data.num_support] for key, value in batch.items()
            }
            batch_query = {
                key: value[self.params.data.num_support:] for key, value in batch.items()
            }
            return  batch_support,batch_query
        else:
            assert isinstance(self.data_iters_support,dict)
            assert isinstance(self.data_iters_query,dict)
            keys_support = list(self.data_iters_support.keys())
            keys_query = list(self.data_iters_query.keys())
            assert len(keys_support) == len(keys_query)
            keys_idx = np.random.choice(np.arange(len(keys_support)),size=1)[0]
            key_support = keys_support[keys_idx]
            key_query =  keys_query[keys_idx]
            try:
                batch_support = next(self.data_iters_support[key_support])
                assert len(batch_support['task']) == self.params.data.num_support
            except (StopIteration, AssertionError):
                # logger.info(f"Reached end of dataloader...")
                self.data_iters_support[key_support] = iter(self.dataloaders_support[key_support])
                batch_support = next(self.data_iters_support[key_support])
            try:
                batch_query = next(self.data_iters_query[key_query])
                assert len(batch_query['task']) == self.params.data.num_query
            except (StopIteration, AssertionError):
                # logger.info(f"Reached end of dataloader...")
                self.data_iters_query[key_query] = iter(self.dataloaders_query[key_query])
                batch_query = next(self.data_iters_query[key_query])
            return batch_support,batch_query



    def data_loss_fn(self, data_output, data_label, data_mask, weight=None):
        loss = F.mse_loss(data_output, data_label, reduction="none")
        pred_mask = data_mask.expand_as(loss)
        if weight is None:
            # no re-weighting, loss is just regular MSE
            loss = (loss * pred_mask).sum() / (pred_mask.sum())
        else:
            # reweight by weight
            weight = weight.expand_as(loss)
            loss = ((loss * pred_mask) * weight).sum()
        return loss
    def prepare_data_deepo(self, samples,train=True):
        data = samples["data"]
        data_mask = samples["data_mask"][:, None, None, :]  # (bs, 1, 1, dim)
        query_locations = samples["query"]



        input_len = self.params.data.input_len
        input_step = self.params.data.input_step
        output_step = self.params.data.output_step
        if train:
            output_start = input_len if self.params.data.output_start is None else self.params.data.output_start
        else:
            output_start = input_len if self.params.data.output_start_eval is None else self.params.data.output_start_eval
        data_label = data[:,output_start::output_step,:]

        # Example of data_input and sensors
        data_input = data[:,:input_len:input_step, :]

        data_input, data_label,data_mask,query_locations = to_cuda(
            (data_input, data_label,data_mask,query_locations)
        )







        if self.params.normalize:
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
        if self.params.loss_weight is None or self.params.loss_weight == "none":
            loss_weight = None
        elif self.params.loss_weight == "l2":
            # reweight by L2 norm squared
            loss_weight = torch.linalg.vector_norm(data_label, dim=(1, 2), keepdim=True) ** 2  # (bs, 1, 1, dim)
        elif self.params.loss_weight == "linfty":
            # reweight by L-infinity norm
            loss_weight, _ = torch.max(torch.abs(data_label), dim=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
        else:
            assert False, f"Unknown reweight type: {self.params.loss_weight}"

        if loss_weight is not None:
            bs = np.single(data_label.size(0))
            # loss_weight = to_cuda(
            #     (torch.reciprocal(loss_weight + eps) / bs).expand_as(data_label).float()
            # )  # (bs, output_len, x_num, x_num, dim)
            loss_weight = to_cuda((torch.reciprocal(loss_weight + eps) / bs).float())  # (bs, 1,  1, dim)

        data_input_reshaped = data_input.repeat_interleave(query_locations.shape[1], dim=0)
        query_tensor_reshaped = query_locations.view(-1,1, 2)
        result_tensor_reshaped = data_label.view(-1, 1,1)
        dict ={
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
    def prepare_data(self, samples, train = True):
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

        symbol_input = symbol[:,1:-1] # Deleting EOS/BOS
        symbol_mask = samples["tree_mask"][:,1:-1]

        input_len = self.params.data.input_len
        input_step = self.params.data.input_step
        output_step = self.params.data.output_step
        if train:
            output_start = input_len if self.params.data.output_start is None else self.params.data.output_start
        else:
            output_start = input_len if self.params.data.output_start_eval is None else self.params.data.output_start_eval


        data_input = data[:, :input_len:input_step]  # (bs, input_len, x_num,  dim)
        data_label = data[:, output_start::output_step]  # (bs, output_len, x_num,  dim)
        input_times = t[:, :input_len:input_step]  # (bs, input_len)
        output_times = t[:, output_start::output_step]  # (bs, output_len)

        data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask = to_cuda(
            (data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask)
        )

        if self.params.normalize:
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
        if self.params.loss_weight is None or self.params.loss_weight == "none":
            loss_weight = None
        elif self.params.loss_weight == "l2":
            # reweight by L2 norm squared
            loss_weight = torch.linalg.vector_norm(data_label, dim=(1, 2), keepdim=True) ** 2  # (bs, 1, 1, dim)
        elif self.params.loss_weight == "linfty":
            # reweight by L-infinity norm
            loss_weight, _ = torch.max(torch.abs(data_label), dim=(1, 2), keepdim=True)  # (bs, 1, 1, dim)
        else:
            assert False, f"Unknown reweight type: {self.params.loss_weight}"

        if loss_weight is not None:
            bs = np.single(data_label.size(0))
            # loss_weight = to_cuda(
            #     (torch.reciprocal(loss_weight + eps) / bs).expand_as(data_label).float()
            # )  # (bs, output_len, x_num, x_num, dim)
            loss_weight = to_cuda((torch.reciprocal(loss_weight + eps) / bs).float())  # (bs, 1,  1, dim)

        dict ={
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

    # def prepare_meta_data(self,samples,train = True):
    #
    #     """
    #     Prepare data for training. (Split entire sequence into input and output, generate loss mask, move to cuda, etc.)
    #
    #     samples: data:         Tensor     (num_task(bs), num_data, max_len, x_num,  dim)
    #              data_mask:    BoolTensor (num_task(bs), dim)
    #              t:            Tensor     (num_task(bs), num_data, max_len)
    #
    #     """
    #
    #     data = samples["data"]
    #     data_mask = samples["data_mask"][:, None, None, None, :]  # (bs, 1, 1, 1, dim)
    #     t = samples["t"]
    #     symbol = samples["tree_encoded"]
    #
    #     symbol_input = [sym[:, 1:-1] for sym in symbol]  # Deleting EOS/BOS
    #     symbol_mask = [mask[:, 1:-1] for mask in samples["tree_mask"]]
    #
    #     input_len = self.params.input_len
    #     input_step = self.params.input_step
    #     output_step = self.params.output_step
    #     if train or output_step == 1:
    #         output_start = input_len
    #     else:
    #         output_start = input_len  + 1
    #     data_input = data[:, :, :input_len:input_step]  # (bs, num_data,input_len, x_num,  dim)
    #     data_label = data[:, :, output_start::output_step]  # (bs, num_data, output_len, x_num,  dim)
    #     input_times = t[:,:, :input_len:input_step]  # (bs, num_data,input_len)
    #     output_times = t[:,:, output_start::output_step]  # (bs, num_data,output_len)
    #
    #     data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask = to_cuda(
    #         (data_input, data_label, input_times, output_times, data_mask, symbol_input, symbol_mask)
    #     )
    #
    #     if self.params.normalize:
    #         mean = torch.mean(data_input, axis=(2,3), keepdim=True)  # (bs, num_data, 1, 1, dim)
    #         std = torch.std(data_input, axis=(2, 3), keepdim=True)  # (bs, num_data,1, 1, dim)
    #
    #         eps = 1e-6
    #         data_input = (data_input - mean) / (std + eps)
    #         data_label = (data_label - mean) / (std + eps)  # use same mean and std
    #
    #     else:
    #         mean = 0
    #         std = 1
    #
    #     # prepare weights for loss function
    #     eps = 1e-5
    #     if self.params.loss_weight is None or self.params.loss_weight == "none":
    #         loss_weight = None
    #     elif self.params.loss_weight == "l2":
    #         # reweight by L2 norm squared
    #         loss_weight = torch.linalg.vector_norm(data_label, dim=(2, 3),
    #                                                keepdim=True) ** 2  # (bs, num_data, 1, 1, dim)
    #     elif self.params.loss_weight == "linfty":
    #         # reweight by L-infinity norm
    #         loss_weight, _ = torch.max(torch.abs(data_label), dim=(2, 3),
    #                                    keepdim=True)  # (bs, num_data, 1, 1, dim)
    #     else:
    #         assert False, f"Unknown reweight type: {self.params.loss_weight}"
    #
    #     if loss_weight is not None:
    #         bs = np.single(data_label.size(0))
    #         # loss_weight = to_cuda(
    #         #     (torch.reciprocal(loss_weight + eps) / bs).expand_as(data_label).float()
    #         # )  # (bs, output_len, x_num, x_num, dim)
    #         loss_weight = to_cuda(
    #             (torch.reciprocal(loss_weight + eps) / bs).float())  # (bs, num_data, 1,  1, dim)
    #
    #     dict = {
    #         "data_input": data_input,
    #         "data_label": data_label,
    #         "mean": mean,
    #         "std": std,
    #         "input_times": input_times,
    #         "output_times": output_times,
    #         "data_mask": data_mask,
    #         "loss_weight": loss_weight,
    #         "symbol_input": symbol_input,
    #         "symbol_mask": symbol_mask
    #     }
    #
    #     return dict

    def iter(self):
        """
        One training step.
        """
        params = self.params


        model = self.modules["model"]
        model.train()

        if params.model.name == "DeepONet":
            # assert not params.meta, "Set meta=0 for DeepONet, not implemented"
            # self.deeponet_update(model)
            self.full_meta_deeponet_updates(model) if params.meta else self.deeponet_update(model)
        elif params.model.name == "FNO":
            # assert not params.meta, "Set meta=0 for FNO, not implemented"
            # self.fno_update(model)
            self.full_meta_fno_updates(model) if params.meta else self.fno_update(model)
        else:
            if  not params.meta:

                # prepare data part
                samples = self.get_batch()

                dict= self.prepare_data(
                    samples
                )

                # forward / loss

                """
                Model input:
                    data_input:   (bs, input_len, x_num, data_dim)
                    input_times:  (bs, input_len, 1)
                    output_times: (bs, output_len, 1)
        
                Model output:
                    data_output:  (bs, output_len, x_num, data_dim)
                """

                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output = model(
                        "fwd",
                        data_input=dict["data_input"],
                        input_times=dict["input_times"][..., None],
                        output_times=dict["output_times"][..., None],
                        symbol_input=dict["symbol_input"],
                        symbol_padding_mask=dict["symbol_mask"],
                    )  # (bs, output_len, x_num, data_dim)
                    data_output = output["data_output"]
                    data_loss = self.data_loss_fn(data_output, dict["data_label"], dict["data_mask"], dict["loss_weight"])

                self.data_loss += data_loss.item()

                # optimize
                self.optimize(data_loss)
            else:
                if self.params.model.name == "prose_freeze_encoder":
                    self.freeze_meta_updates_encoder(model)
                # elif self.params.model.name == "prose_freezed_symbol":
                #     self.freeze_meta_updates_symbol(model)
                else:
                    self.full_meta_updates(model)


        self.inner_epoch += 1
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def deeponet_update(self,model):
        params = self.params
        samples = self.get_batch()

        dict = self.prepare_data_deepo(
            samples
        )
        with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
            output = model(
                querypoint = dict["query_tensor_reshaped"],
                value_at_sensor= dict["data_input_reshaped"],
            )  # (bs, output_len, x_num, data_dim)
            output_start = self.params.data.input_len if  self.params.data.output_start is None else self.params.data.output_start
            num_output_t =  (params.data.t_num - output_start + 1)//self.params.data.output_step
            data_output = output.reshape(params.batch_size,num_output_t,params.data.x_num,1)
            data_loss = self.data_loss_fn(data_output, dict["data_label"], dict["data_mask"], dict["loss_weight"])

        self.data_loss += data_loss.item()

        # optimize
        self.optimize(data_loss)
    def full_meta_deeponet_updates(self,model):
        params = self.params
        query_data_loss = to_cuda(torch.tensor(0.0).float())
        for ii in range(params.batch_size_task):
            samples_support, samples_query = self.get_task()
            dict_support = self.prepare_data_deepo(
                samples_support
            )
            dict_query = self.prepare_data_deepo(
                samples_query
            )

            learner = model.clone()
            learner = to_cuda(learner)

            for _ in range(params.meta_step):
                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output = learner(
                    querypoint = dict_support["query_tensor_reshaped"],
                    value_at_sensor= dict_support["data_input_reshaped"],
                        )
                    output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
                    num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                    data_output = output.reshape(params.data.num_support, num_output_t, params.data.x_num, 1)
                    support_data_loss = self.data_loss_fn(data_output, dict_support["data_label"],dict_support["data_mask"],dict_support["loss_weight"])

                learner.adapt(support_data_loss)
            learner.eval()

            with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                output = learner(
                    querypoint=dict_query["query_tensor_reshaped"],
                    value_at_sensor=dict_query["data_input_reshaped"],
                )
                output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
                num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                data_output = output.reshape(params.data.num_query, num_output_t, params.data.x_num, 1)
                query_data_loss += self.data_loss_fn(data_output, dict_query["data_label"],
                                                      dict_query["data_mask"], dict_query["loss_weight"])


        query_data_loss /= params.batch_size
        self.data_loss += query_data_loss.item()
        # optimize
        self.optimizer.zero_grad()
        self.optimize(query_data_loss)
    def fno_update(self,model):
        params = self.params
        samples = self.get_batch()

        dict = self.prepare_data(
            samples
        )

        output = model(
           dict["data_input"],
        )  # (bs, output_len, x_num, data_dim)
        output_start = self.params.data.input_len if  self.params.data.output_start is None else self.params.data.output_start
        num_output_t =  (params.data.t_num - output_start + 1)//self.params.data.output_step
        data_output = output.reshape(params.batch_size,num_output_t,params.data.x_num,1)
        data_loss = self.data_loss_fn(data_output, dict["data_label"], dict["data_mask"], dict["loss_weight"])

        self.data_loss += data_loss.item()

        # optimize
        self.optimize(data_loss)


    def full_meta_fno_updates(self,model):
        params = self.params
        query_data_loss = to_cuda(torch.tensor(0.0).float())
        for ii in range(params.batch_size_task):
            samples_support, samples_query = self.get_task()
            dict_support = self.prepare_data(
                samples_support
            )
            dict_query = self.prepare_data(
                samples_query
            )

            learner = model.clone()
            learner = to_cuda(learner)

            for _ in range(params.meta_step):
                # with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                output = learner(
                    dict_support["data_input"],
                )  # (bs, output_len, x_num, data_dim)
                output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
                num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
                data_output = output.reshape(params.data.num_support, num_output_t, params.data.x_num, 1)
                support_data_loss = self.data_loss_fn(data_output, dict_support["data_label"], dict_support["data_mask"],dict_support["loss_weight"])
            learner.adapt(support_data_loss)
            learner.eval()

            output = learner(
                dict_query["data_input"],
            )  # (bs, output_len, x_num, data_dim)
            output_start = self.params.data.input_len if self.params.data.output_start is None else self.params.data.output_start
            num_output_t = (params.data.t_num - output_start + 1) // self.params.data.output_step
            data_output = output.reshape(params.data.num_query, num_output_t, params.data.x_num, 1)
            query_data_loss += self.data_loss_fn(data_output, dict_query["data_label"], dict_query["data_mask"],dict_query["loss_weight"])


        query_data_loss /= params.batch_size
        self.data_loss += query_data_loss.item()
        # optimize
        self.optimizer.zero_grad()
        self.optimize(query_data_loss)
    # def full_meta_oneupdate(self,params, model, dict_supports, dict_querys, data_loss_fn, to_cuda, ii):
    #     dict_support = dict_supports[ii]
    #     dict_query = dict_querys[ii]
    #
    #     learner = model.clone()
    #     learner = to_cuda(learner)
    #
    #     for _ in range(params.meta_step):
    #         with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
    #             output_dict = learner(
    #                 "fwd",
    #                 data_input=dict_support["data_input"],
    #                 input_times=dict_support["input_times"][..., None],
    #                 output_times=dict_support["output_times"][..., None],
    #                 symbol_input=dict_support["symbol_input"],
    #                 symbol_padding_mask=dict_support["symbol_mask"],
    #             )
    #             data_output = output_dict["data_output"]
    #             support_data_loss = data_loss_fn(data_output, dict_support["data_label"],
    #                                              dict_support["data_mask"],
    #                                              dict_support["loss_weight"])
    #             learner.adapt(support_data_loss)
    #
    #     learner.eval()
    #     with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
    #         output_dict = learner(
    #             "fwd",
    #             data_input=dict_query["data_input"],
    #             input_times=dict_query["input_times"][..., None],
    #             output_times=dict_query["output_times"][..., None],
    #             symbol_input=dict_query["symbol_input"],
    #             symbol_padding_mask=dict_query["symbol_mask"],
    #         )  # (bs, output_len, x_num, data_dim)
    #         data_output = output_dict["data_output"]
    #         query_data_loss = data_loss_fn(data_output, dict_query["data_label"],
    #                                        dict_query["data_mask"],
    #                                        dict_query["loss_weight"])
    #     return query_data_loss
    def full_meta_updates(self,model):
        params = self.params
        query_data_loss = to_cuda(torch.tensor(0.0).float())
        if not self.params.model.meta.first_order:
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [torch.nn.attention.SDPBackend.MATH,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION]

        with torch.nn.attention.sdpa_kernel(backends):
            for ii in range(params.batch_size_task):
                samples_support, samples_query = self.get_task()
                dict_support = self.prepare_data(
                    samples_support
                )
                dict_query = self.prepare_data(
                    samples_query
                )

                learner = model.clone()
                learner = to_cuda(learner)

                for _ in range(params.meta_step):
                    with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                        output_dict = learner(
                            "fwd",
                            data_input=dict_support["data_input"],
                            input_times=dict_support["input_times"][..., None],
                            output_times=dict_support["output_times"][..., None],
                            symbol_input=dict_support["symbol_input"],
                            symbol_padding_mask=dict_support["symbol_mask"],
                        )
                        data_output = output_dict["data_output"]
                        support_data_loss = self.data_loss_fn(data_output, dict_support["data_label"],
                                                              dict_support["data_mask"],
                                                              dict_support["loss_weight"])
                        learner.adapt(support_data_loss)
                learner.eval()

                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output_dict = learner(
                        "fwd",
                        data_input=dict_query["data_input"],
                        input_times=dict_query["input_times"][..., None],
                        output_times=dict_query["output_times"][..., None],
                        symbol_input=dict_query["symbol_input"],
                        symbol_padding_mask=dict_query["symbol_mask"],
                    )  # (bs, output_len, x_num, data_dim)
                    data_output = output_dict["data_output"]
                    query_data_loss += self.data_loss_fn(data_output, dict_query["data_label"],
                                                         dict_query["data_mask"],
                                                         dict_query["loss_weight"])

            '''
            dict_supports = []
            dict_querys = []
            for ii in range(params.batch_size):
                samples_support, samples_query = self.get_task()
                dict_support = self.prepare_data(samples_support)
                dict_query = self.prepare_data(samples_query)
                dict_supports.append(dict_support)
                dict_querys.append(dict_query)

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(self.full_meta_oneupdate, params, model,dict_supports, dict_querys, self.data_loss_fn, to_cuda, ii)
                    for ii in range(params.batch_size)]

                for future in concurrent.futures.as_completed(futures):
                    query_data_loss += future.result()
            '''
            query_data_loss /= params.batch_size
            self.data_loss += query_data_loss.item()
            # optimize
            self.optimizer.zero_grad()
            self.optimize(query_data_loss)


    def freeze_meta_updates_encoder(self,model):
        params = self.params
        query_data_loss = to_cuda(torch.tensor(0.0).float())
        if not self.params.model.meta.first_order:
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [torch.nn.attention.SDPBackend.MATH,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION]

        with torch.nn.attention.sdpa_kernel(backends):
            for ii in range(params.batch_size):
                samples_support, samples_query = self.get_task()
                dict_support = self.prepare_data(
                    samples_support
                )
                dict_query = self.prepare_data(
                    samples_query
                )

                learner = model.inner_model.clone()
                learner = to_cuda(learner)
                output_noinner = model.no_inner_model(
                    "fwd",
                    data_input= dict_support["data_input"],
                    input_times= dict_support["input_times"][..., None],
                    symbol_input=dict_support["symbol_input"],
                    symbol_padding_mask=dict_support["symbol_mask"],)

                for _ in range(params.meta_step):
                    with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                        output_dict = learner(
                            "fwd",
                            output_times=dict_support["output_times"][..., None],
                            embedder=model.no_inner_model.embedder,
                            fused=output_noinner["fused"],
                            fused_mask=output_noinner["fused_mask"]
                        )
                        data_output = output_dict["data_output"]
                        support_data_loss = self.data_loss_fn(data_output, dict_support["data_label"],
                                                              dict_support["data_mask"],
                                                              dict["loss_weight"])
                        learner.adapt(support_data_loss)
                learner.eval()
                output_noinner = model.no_inner_model( "fwd",
                                                       data_input=dict_query["data_input"],
                                                       input_times=dict_query["input_times"][ ..., None],
                                                       symbol_input=dict_query["symbol_input"],
                                                       symbol_padding_mask=dict_query["symbol_mask"])
                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output_dict = learner(
                        "fwd",
                        output_times=dict_query["output_times"][..., None],
                        embedder=model.no_inner_model.embedder,
                        fused=output_noinner["fused"],
                        fused_mask=output_noinner["fused_mask"]
                    )  # (bs, output_len, x_num, data_dim)
                    data_output = output_dict["data_output"]
                    query_data_loss += self.data_loss_fn(data_output, dict_query["data_label"],
                                                         dict_query["data_mask"],
                                                         dict["loss_weight"])

            query_data_loss /= params.batch_size
            self.data_loss += query_data_loss.item()
            # optimize
            self.optimizer.zero_grad()
            self.optimize(query_data_loss)

    def freeze_meta_updates_symbol(self,model):
        params = self.params
        query_data_loss = to_cuda(torch.tensor(0.0).float())
        if not self.params.model.meta.first_order:
            backends = [torch.nn.attention.SDPBackend.MATH]
        else:
            backends = [torch.nn.attention.SDPBackend.MATH,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION]

        with torch.nn.attention.sdpa_kernel(backends):
            for ii in range(params.batch_size):
                samples_support, samples_query = self.get_task()
                dict_support = self.prepare_data(
                    samples_support
                )
                dict_query = self.prepare_data(
                    samples_query
                )
                learner = model.inner_model.clone()
                learner = to_cuda(learner)
                output_noinner = model.no_inner_model(
                    "fwd",
                    symbol_input=dict_support["symbol_input"],
                    symbol_padding_mask=dict_support["symbol_mask"]
                )

                for _ in range(params.meta_step):
                    with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                        output_dict = learner(
                            "fwd",
                            data_input=dict_support["data_input"],
                            input_times=dict_support["input_times"][..., None],
                            output_times=dict_support["output_times"][..., None],
                            symbol_encoded=output_noinner["symbol_encoded"],
                            symbol_padding_mask=dict_support["symbol_mask"],
                        )
                        data_output = output_dict["data_output"]
                        support_data_loss = self.data_loss_fn(data_output, dict_support["data_label"],
                                                              dict_support["data_mask"],
                                                              dict_support["loss_weight"])
                        learner.adapt(support_data_loss)
                learner.eval()
                output_noinner = model.no_inner_model( "fwd",
                                                       symbol_input=dict_query["symbol_input"],
                                                       symbol_padding_mask=dict_query["symbol_mask"]
                                                       )
                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output_dict = learner(
                        "fwd",
                        data_input=dict_query["data_input"],
                        input_times=dict_query["input_times"][..., None],
                        output_times=dict_query["output_times"][..., None],
                        symbol_encoded=output_noinner["symbol_encoded"],
                        symbol_padding_mask=dict_query["symbol_mask"],
                    )  # (bs, output_len, x_num, data_dim)
                    data_output = output_dict["data_output"]
                    query_data_loss += self.data_loss_fn(data_output, dict_query["data_label"],
                                                         dict_query["data_mask"],
                                                         dict_query["loss_weight"])

            query_data_loss /= params.batch_size
            self.data_loss += query_data_loss.item()
            # optimize
            self.optimizer.zero_grad()
            self.optimize(query_data_loss)