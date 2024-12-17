from logging import getLogger
import torch
from tabulate import tabulate
from collections import OrderedDict
from .transformer_wrappers import  PROSE_1DPDE, Combine_freeze_encoder, PROSE_1DPDE_inner_data,PROSE_1DPDE_freeze_symbol_encoder
from other_models.deeponet import DeepONet
from .meta_model import MAML, MetaSGD, MAMLAdamW
from .finetune_model import assign_linear_lora
logger = getLogger()
from neuralop.models import FNO
import numpy as np

def reload_module(modules,reloaded):

    def normalize_key(key):
        prefixes = ["module.", "inner_model.module.", "no_inner_model.", "_orig_mod."]
        for prefix in prefixes:
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    # Assuming modules and reloaded are given
    # modules = {'model_name': model_instance}
    # reloaded = {'model_name': state_dict}

    for k, v in modules.items():
        assert k in reloaded, f"{k} not in save"

        original_keys = reloaded[k].keys()
        transformed_keys = {}

        for key in original_keys:
            new_key = normalize_key(key)
            transformed_keys[new_key] = reloaded[k][key]

        # Load state dict into the model, matching the normalized keys
        model_state_dict = v.state_dict()
        new_model_state_dict = OrderedDict()

        for model_key in model_state_dict.keys():
            norm_model_key = normalize_key(model_key)
            if norm_model_key in transformed_keys:
                new_model_state_dict[model_key] = transformed_keys[norm_model_key]
            else:
                new_model_state_dict[model_key] = model_state_dict[model_key]

        v.load_state_dict(new_model_state_dict)


def build_model(params, model_config, data_config, symbol_env):
    modules = {}

    # get model
    name = model_config.name

    if name == "prose":
        # 2to1 prose model
        base_model = PROSE_1DPDE(
            model_config,
            symbol_env,
            data_config
        )

        if not params.zero_shot_only:
            if model_config.meta.name == "MAML":
                modules["model"] = MAML(base_model,
                                model_config.meta.meta_lr,
                                    eta=model_config.meta.gd_eta,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    clip_norm=params.clip_grad_norm,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MAMLAdamW":
                modules["model"] = MAMLAdamW(base_model,
                                    model_config.meta.meta_lr,
                                    betas=(0.9, 0.999),
                                    eps=params.optim.get("eps", 1e-8),
                                    weight_decay=params.optim.weight_decay,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MetaSGD":
                modules["model"] = MetaSGD(base_model,
                                model_config.meta.meta_lr,
                                    first_order=model_config.meta.first_order)
        else:
            modules["model"] = base_model

    elif name == "prose_freeze_symbol":
        no_inner_model = PROSE_1DPDE_freeze_symbol_encoder(
            model_config,
            symbol_env,
            data_config
        )
        inner_model = PROSE_1DPDE_inner_data(
            model_config,
            symbol_env,
            data_config
        )
        inner_model = MAML(inner_model,
                                    model_config.meta.meta_lr,
                                    eta = model_config.meta.gd_eta,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    allow_unused=model_config.meta.allow_unused)
        if not params.zero_shot_only:
            modules["model"] = Combine_freeze_encoder(no_inner_model,inner_model)
        else:
            modules["model"] = PROSE_1DPDE(
                model_config,
                symbol_env,
                data_config
            )
            params.freezedmodule = ["embedder", "data_encoder", "symbol_encoder", "fusion"]

    elif name == "DeepONet":
        base_model = DeepONet(
            model_config, data_config
        )

        if not params.zero_shot_only:
            if model_config.meta.name == "MAML":
                modules["model"] = MAML(base_model,
                                model_config.meta.meta_lr,
                                    eta=model_config.meta.gd_eta,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    clip_norm=params.clip_grad_norm,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MAMLAdamW":
                modules["model"] = MAMLAdamW(base_model,
                                    model_config.meta.meta_lr,
                                    betas=(0.9, 0.999),
                                    eps=params.optim.get("eps", 1e-8),
                                    weight_decay=params.optim.weight_decay,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MetaSGD":
                modules["model"] = MetaSGD(base_model,
                                model_config.meta.meta_lr,
                                    first_order=model_config.meta.first_order)
        else:
            modules["model"] = base_model
    elif name == "FNO":
        output_start =  data_config.input_len if data_config.output_start is None else data_config.output_start
        output_start_eval = data_config.input_len if data_config.output_start_eval is None else data_config.output_start_eval
        assert output_start // data_config.output_step == output_start_eval // data_config.output_step
        base_model= FNO(
            n_modes=model_config.n_modes, hidden_channels=model_config.hidden_channels,
            in_channels=int(np.ceil(data_config.input_len / data_config.input_step)) , out_channels= int(np.ceil((data_config.t_num-output_start) / data_config.output_step))
        )
        if not params.zero_shot_only:
            if model_config.meta.name == "MAML":
                modules["model"] = MAML(base_model,
                                model_config.meta.meta_lr,
                                    eta=model_config.meta.gd_eta,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                        clip_norm=params.clip_grad_norm,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MAMLAdamW":
                modules["model"] = MAMLAdamW(base_model,
                                    model_config.meta.meta_lr,
                                    betas=(0.9, 0.999),
                                    eps=params.optim.get("eps", 1e-8),
                                    weight_decay=params.optim.weight_decay,
                                    first_order=model_config.meta.first_order,
                                    allow_nograd=model_config.meta.allow_nograd,
                                    allow_unused=model_config.meta.allow_unused)
            elif model_config.meta.name == "MetaSGD":
                modules["model"] = MetaSGD(base_model,
                                model_config.meta.meta_lr,
                                    first_order=model_config.meta.first_order)
        else:
            modules["model"] = base_model
    else:
        assert False, f"Model {name} hasn't been implemented"

    if params.reload_model and not params.reload_ftmodel:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        reload_module(modules,reloaded)

    if params.finetune and params.finetune_name == "lora":
            modules["model"] = assign_linear_lora(modules["model"],params)
    if params.freezedmodule is not None:
            for name, param in modules["model"].named_parameters():
                if name.startswith('module.'):
                    name = name[len('module.'):]
                if name.startswith(tuple(params.freezedmodule)):
                    param.requires_grad = False
    # reload pretrained modules

    if params.reload_ftmodel:
        logger.info(f"Reloading modules from {params.reload_ftmodel} ...")
        reloaded = torch.load(params.reload_ftmodel)
        reload_module(modules,reloaded)


    #
    # if params.reload_ftmodel:
    #     logger.info(f"Reloading modules from {params.reload_ftmodel} ...")
    #     reloaded = torch.load(params.reload_ftmodel)
    #     for k, v in modules.items():
    #         assert k in reloaded, f"{k} not in save"
    #         # if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
    #         #     reloaded[k] = {k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()}
    #         if all([k2.startswith("_orig_mod.") for k2 in reloaded[k].keys()]):
    #             reloaded[k] = {k2[len("_orig_mod.") :]: v2 for k2, v2 in reloaded[k].items()}
    #         v.load_state_dict(reloaded[k])
    # log
    for k, v in modules.items():
        logger.info(f"{k}: {v}")
    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad]):,}"
        if hasattr(v, "summary"):
            # for individual components of a wrapper model
            s += v.summary()
        logger.info(s)

    # for k, v in modules.items():
    #     table_data = [(name, str(param.shape), param.requires_grad) for name, param in v.named_parameters()]
    #     logger.info("\n" + tabulate(table_data, headers=["Parameter Name", "Shape", "Requires Grad"], tablefmt="grid"))
    #     table_data = [(name, str(param.shape)) for name, param in v.named_parameters() if param.requires_grad]
    #     logger.info("\n" + tabulate(table_data, headers=["Trainable Parameters", "Shape"], tablefmt="grid"))

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    if params.compile:
        for k, v in modules.items():
            # modules[k] = torch.compile(v, mode="reduce-overhead")
            modules[k] = torch.compile(v)

    return modules
