import torch
from functools import partial
import copy

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

def assign_linear_lora(model, params):
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=params.lora_r, alpha=params.lora_alpha)
    # freezedmodule=["embedder", "data_encoder", ...]
    # To modify modules, keep track of parent modules and their attribute names
    modules_to_update = []
    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear) and  (params.freezedmodule is None or not name.startswith(tuple(params.freezedmodule))):
            parent, attr_name = name.rsplit('.', 1)
            parent_module = dict(model.named_modules())[parent]
            modules_to_update.append((parent_module, attr_name, module))

    # Update the modules with new ones
    for parent_module, attr_name, module in modules_to_update:
        setattr(parent_module, attr_name, assign_lora(module))

    return model

# if __name__ == '__main__':
#     import hydra
#     import sys
#
#     sys.path.append('/home/jingmins/matryoshka/src')
#     from models.transformer_wrappers import PROSE_Fluids, PROSE_1DPDE, PROSE_1DPDE_freeze_data, \
#         PROSE_1DPDE_inner_data, Combine_freeze_data, Combine_freeze_symbol, \
#         PROSE_1DPDE_freeze_symbol, PROSE_1DPDE_inner_symbol
#     from models.meta_model import MAML, MetaSGD
#     from symbol_utils.environment import SymbolicEnvironment
#
#
#     @hydra.main(version_base=None, config_path="../configs", config_name="main")
#     def test(params):
#         params.lora_r=10
#         params.lora_alpha=0.01
#         symbol_env = SymbolicEnvironment(params.symbol)
#         # 2to1 prose model
#         model_config = params.model
#         base_model = PROSE_1DPDE(
#             model_config, symbol_env,params.data,
#         )
#         modules = {}
#         ## Only consider MAML, if regular training (No Meta updates), we still use MAML() framwork, but no meta updates, just regular updates, check trainer.py for details
#         modules["model"] = MAML(base_model,
#                             model_config.meta.meta_lr,
#                                 eta=model_config.meta.gd_eta,
#                                 first_order=model_config.meta.first_order,
#                                 allow_nograd=model_config.meta.allow_nograd,
#                                 allow_unused=model_config.meta.allow_unused)
#         modules["model_lora"] = assign_linear_lora(modules["model"],params)
#
#         for k, v in modules.items():
#             print(f"{k}: {v}")
#         for k, v in modules.items():
#             s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad]):,}"
#             if hasattr(v, "summary"):
#                 # for individual components of a wrapper model
#                 s += v.summary()
#             print(s)
#
#     test()