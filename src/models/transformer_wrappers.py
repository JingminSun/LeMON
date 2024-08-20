"""
Final model wrappers. 
"""

import torch
import torch.nn as nn

from .attention_utils import get_padding_mask
from .transformer import (
    TransformerDataEncoder,
    DataOperatorDecoder,
    TransformerSymbolEncoder,
    TransformerFusion,
)
from .embedder import LinearEmbedder, LinearEmbedder_1DPDE
from logging import getLogger
from .meta_utils import freeze, unfreeze

logger = getLogger()


class PROSE_1DPDE(nn.Module):
    """
    Wrapper for the full PROSE model (2to1).
    For 1D PDE
    """

    def __init__(self, config, symbol_env, data_config):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = data_config.x_num
        self.max_output_dim = data_config.max_output_dimension

        self.embedder = LinearEmbedder_1DPDE(config.embedder, self.x_num, self.max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        if not self.config.no_text:
            self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
            self.fusion = TransformerFusion(config.fusion)
        self.data_decoder = DataOperatorDecoder(config.data_decoder)


    # def freeze_param(self, freeze_module):
    #     if not isinstance(freeze_module,list):
    #         freeze_module = [freeze_module]
    #     for module in freeze_module:
    #         if module == "embedder":
    #             self.embedder = freeze(self.embedder)
    #         elif module == "data_encoder":
    #             self.data_encoder = freeze(self.data_encoder)
    #         elif module == "symbol_encoder":
    #             self.symbol_encoder = freeze(self.symbol_encoder)
    #         elif module == "fusion":
    #             self.fusion =freeze(self.fusion)
    #         elif module == "data_decoder":
    #             self.data_decoder = freeze(self.ddata_decoder)
    #         else:
    #             raise f"Module {module} not in the model, unable to freeze"
    #
    # def unfreeze_param(self, unfreeze_module):
    #     if not isinstance(unfreeze_module,list):
    #         unfreeze_module = [unfreeze_module]
    #     for module in unfreeze_module:
    #         if module == "embedder":
    #             self.embedder = unfreeze(self.embedder)
    #         elif module == "data_encoder":
    #             self.data_encoder = unfreeze(self.data_encoder)
    #         elif module == "symbol_encoder":
    #             self.symbol_encoder = unfreeze(self.symbol_encoder)
    #         elif module == "fusion":
    #             self.fusion =unfreeze(self.fusion)
    #         elif module == "data_decoder":
    #             self.data_decoder = unfreeze(self.ddata_decoder)
    #         else:
    #             raise f"Module {module} not in the model, unable to unfreeze"

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
        if not self.config.no_text:
            s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
            s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        data_input,
        input_times,
        output_times,
        symbol_input,
        symbol_padding_mask=None,
    ):
        """
        Inputs:
            data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:            Tensor     (bs, input_len, 1)
            output_times:           Tensor     (bs, output_len, 1)

            symbol_input:           LongTensor           (bs, symbol_len)
            symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        output = {}
        bs, input_len, x_num,  data_dim = data_input.size()
        # symbol_len = symbol_input.size(1)
        # symbol_padding_mask = get_padding_mask(symbol_lengths)  # (bs, max_len)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num 
        """

        data_input = self.embedder.encode(data_input, input_times)  # (bs, data_len, dim)
        data_len = data_input.size(1)
        output["data_embeded"] = data_input
        """
        Step 2: Encode + Fusion
            data_input:   Tensor     (bs, data_len, dim)
            symbol_input: LongTensor (bs, symbol_len)
        """

        data_encoded = self.data_encoder(data_input)  # (bs, data_len, dim)
        if not self.config.no_text:
            symbol_encoded = self.symbol_encoder(
                symbol_input, src_key_padding_mask=symbol_padding_mask
            )  # (bs, symbol_len, dim)
            fused, fused_mask = self.fusion(
                x0=data_encoded,
                x1=symbol_encoded,
                key_padding_mask0=None,
                key_padding_mask1=symbol_padding_mask,
            )  # (bs, data_len+symbol_len, dim)
        else:
            symbol_encoded = None
            fused = data_encoded
            fused_mask = None

        output["data_encoded"] = data_encoded
        output["symbol_encoded"] = symbol_encoded
        output["fused"] = fused
        """
        Step 3: Decode data
        """

        query_emb = self.data_decoder.get_query_emb(output_times)  # (bs, query_len, dim)

        data_output = self.data_decoder(
            src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
        )  # (bs, query_len, dim)

        data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)

        output["data_output"] = data_output


        return output

    def generate(self, **kwargs):
        return self.fwd(**kwargs)

class PROSE_1DPDE_inner_data_decoder(nn.Module):
    """
    Wrapper for the full PROSE model (2to1).
    For 1D PDE
    """

    def __init__(self, config, symbol_env, data_config):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = data_config.x_num
        self.max_output_dim = data_config.max_output_dimension

        # self.embedder = LinearEmbedder_1DPDE(config.embedder, self.x_num, self.max_output_dim)
        # self.data_encoder = TransformerDataEncoder(config.data_encoder)
        # self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
        # self.fusion = TransformerFusion(config.fusion)
        self.data_decoder = DataOperatorDecoder(config.data_decoder)


    def summary(self):
        s = "\n"
        # s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        # s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
        # s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
        # s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        output_times,
        embedder,
        fused,
        fused_mask
    ):
        """
        Inputs:
            data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:            Tensor     (bs, input_len, 1)
            output_times:           Tensor     (bs, output_len, 1)

            symbol_input:           LongTensor           (bs, symbol_len)
            symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        output = {}

        query_emb = self.data_decoder.get_query_emb(output_times)  # (bs, query_len, dim)

        data_output = self.data_decoder(
            src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
        )  # (bs, query_len, dim)

        data_output = embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)

        output["data_output"] = data_output


        return output

    def generate(self, **kwargs):
        return self.fwd(**kwargs)

class PROSE_1DPDE_freeze_encoder(nn.Module):
    """
    Wrapper for the full PROSE model (2to1).
    For 1D PDE
    """

    def __init__(self, config, symbol_env, data_config):
        super().__init__()
        self.config = config
        self.symbol_env = symbol_env
        self.x_num = data_config.x_num
        self.max_output_dim = data_config.max_output_dimension

        self.embedder = LinearEmbedder_1DPDE(config.embedder, self.x_num, self.max_output_dim)
        self.data_encoder = TransformerDataEncoder(config.data_encoder)
        self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
        self.fusion = TransformerFusion(config.fusion)
    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
        s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
        s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == "fwd":
            return self.fwd(**kwargs)
        elif mode == "generate":
            return self.generate(**kwargs)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(
        self,
        data_input,
        input_times,
        symbol_input,
        symbol_padding_mask = None,
    ):
        """
        Inputs:
            data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
            input_times:            Tensor     (bs, input_len, 1)
            output_times:           Tensor     (bs, output_len, 1)

            symbol_input:           LongTensor           (bs, symbol_len)
            symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """
        output = {}
        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, input_len, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = input_len * patch_num 
        """

        data_input = self.embedder.encode(data_input, input_times)  # (bs, data_len, dim)
        data_len = data_input.size(1)
        output["data_embeded"] = data_input
        """
        Step 2: Encode + Fusion
            data_input:   Tensor     (bs, data_len, dim)
            symbol_input: LongTensor (bs, symbol_len)
        """

        data_encoded = self.data_encoder(data_input)  # (bs, data_len, dim)
        output["data_encoded"] = data_encoded
        symbol_encoded = self.symbol_encoder(
            symbol_input, src_key_padding_mask=symbol_padding_mask
        )  # (bs, symbol_len, dim)
        fused, fused_mask = self.fusion(
            x0=data_encoded,
            x1=symbol_encoded,
            key_padding_mask0=None,
            key_padding_mask1=symbol_padding_mask,
        )  # (bs, data_len+symbol_len, dim)
        output["fused"] = fused
        output["fused_mask"] = fused_mask
        return output

    def generate(self, **kwargs):
        return self.fwd(**kwargs)

# class PROSE_1DPDE_inner_symbol(nn.Module):
#     """
#     Wrapper for the full PROSE model (2to1).
#     For 1D PDE
#     """
#
#     def __init__(self, config, symbol_env, data_config):
#         super().__init__()
#         self.config = config
#         self.symbol_env = symbol_env
#         self.x_num = data_config.x_num
#         self.max_output_dim = data_config.max_output_dimension
#
#         self.embedder = LinearEmbedder_1DPDE(config.embedder, self.x_num, self.max_output_dim)
#         self.data_encoder = TransformerDataEncoder(config.data_encoder)
#         # self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
#         self.fusion = TransformerFusion(config.fusion)
#         self.data_decoder = DataOperatorDecoder(config.data_decoder)
#
#
#     def summary(self):
#         s = "\n"
#         s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
#         s += f"\tData Encoder:    {sum([p.numel() for p in self.data_encoder.parameters() if p.requires_grad]):,}\n"
#         # s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
#         s += f"\tFusion:          {sum([p.numel() for p in self.fusion.parameters() if p.requires_grad]):,}\n"
#         s += f"\tData Decoder:    {sum([p.numel() for p in self.data_decoder.parameters() if p.requires_grad]):,}"
#         return s
#
#     def forward(self, mode, **kwargs):
#         """
#         Forward function with different forward modes.
#         ### Small hack to handle PyTorch distributed.
#         """
#         if mode == "fwd":
#             return self.fwd(**kwargs)
#         elif mode == "generate":
#             return self.generate(**kwargs)
#         else:
#             raise Exception(f"Unknown mode: {mode}")
#
#     def fwd(
#         self,
#         data_input,
#         input_times,
#         output_times,
#         symbol_encoded,
#         symbol_padding_mask=None,
#     ):
#         """
#         Inputs:
#             data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
#             input_times:            Tensor     (bs, input_len, 1)
#             output_times:           Tensor     (bs, output_len, 1)
#
#             symbol_input:           LongTensor           (bs, symbol_len)
#             symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements
#
#         Output:
#             data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
#         """
#         output = {}
#         """
#         Step 1: Prepare data input (add time embeddings and patch position embeddings)
#             data_input (bs, input_len, x_num, data_dim) -> (bs, data_len, dim)
#                        data_len = input_len * patch_num
#         """
#
#         data_input = self.embedder.encode(data_input, input_times)  # (bs, data_len, dim)
#         data_len = data_input.size(1)
#         output["data_embeded"] = data_input
#         """
#         Step 2: Encode + Fusion
#             data_input:   Tensor     (bs, data_len, dim)
#             symbol_input: LongTensor (bs, symbol_len)
#         """
#
#         data_encoded = self.data_encoder(data_input)  # (bs, data_len, dim)
#         # symbol_encoded = self.symbol_encoder(
#         #     symbol_input, src_key_padding_mask=symbol_padding_mask
#         # )  # (bs, symbol_len, dim)
#
#         fused, fused_mask = self.fusion(
#             x0=data_encoded,
#             x1=symbol_encoded,
#             key_padding_mask0=None,
#             key_padding_mask1=symbol_padding_mask,
#         )  # (bs, data_len+symbol_len, dim)
#
#         output["data_encoded"] = data_encoded
#         output["symbol_encoded"] = symbol_encoded
#         output["fused"] = fused
#         """
#         Step 3: Decode data
#         """
#
#         query_emb = self.data_decoder.get_query_emb(output_times)  # (bs, query_len, dim)
#
#         data_output = self.data_decoder(
#             src=fused, query_emb=query_emb, src_key_padding_mask=fused_mask
#         )  # (bs, query_len, dim)
#
#         data_output = self.embedder.decode(data_output)  # (bs, output_len, x_num, x_num, data_dim)
#
#         output["data_output"] = data_output
#
#
#         return output
#
#     def generate(self, **kwargs):
#         return self.fwd(**kwargs)
#
# class PROSE_1DPDE_freeze_symbol(nn.Module):
#     """
#     Wrapper for the full PROSE model (2to1).
#     For 1D PDE
#     """
#
#     def __init__(self, config, symbol_env, data_config):
#         super().__init__()
#         self.config = config
#         self.symbol_env = symbol_env
#         self.x_num = data_config.x_num
#         self.max_output_dim = data_config.max_output_dimension
#
#         self.symbol_encoder = TransformerSymbolEncoder(config.symbol_encoder, symbol_env.equation_id2word)
#
#
#
#     def summary(self):
#         s = "\n"
#         s += f"\tSymbol Encoder:  {sum([p.numel() for p in self.symbol_encoder.parameters() if p.requires_grad]):,}\n"
#         return s
#
#     def forward(self, mode, **kwargs):
#         """
#         Forward function with different forward modes.
#         ### Small hack to handle PyTorch distributed.
#         """
#         if mode == "fwd":
#             return self.fwd(**kwargs)
#         elif mode == "generate":
#             return self.generate(**kwargs)
#         else:
#             raise Exception(f"Unknown mode: {mode}")
#
#     def fwd(
#         self,
#         symbol_input,
#         symbol_padding_mask = None
#     ):
#         """
#         Inputs:
#             data_input:             Tensor     (bs, input_len, x_num, x_num, data_dim)
#             input_times:            Tensor     (bs, input_len, 1)
#             output_times:           Tensor     (bs, output_len, 1)
#
#             symbol_input:           LongTensor           (bs, symbol_len)
#             symbol_padding_mask:    LongTensor           (bs, symbol_len) # True for padded elements
#
#         Output:
#             data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
#         """
#         output = {}
#         symbol_encoded = self.symbol_encoder(
#             symbol_input, src_key_padding_mask=symbol_padding_mask
#         )  # (bs, symbol_len, dim)
#         output["symbol_encoded"] = symbol_encoded
#
#         return output
#
#     def generate(self, **kwargs):
#         return self.fwd(**kwargs)

class Combine_freeze_encoder(nn.Module):
    def __init__(self, no_inner_model, inner_model):
        super().__init__()
        self.no_inner_model = no_inner_model
        self.inner_model = inner_model

    def forward(self, mode, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
        if mode == "fwd":
            return self.fwd(data_input, input_times, output_times, symbol_input, symbol_padding_mask)
        elif mode == "generate":
            return self.generate(data_input, input_times, output_times, symbol_input, symbol_padding_mask)
        else:
            raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
        output_noinner = self.no_inner_model("fwd",
                                             data_input = data_input,
                                             input_times = input_times,
                                             symbol_input= symbol_input,
                                             symbol_padding_mask=symbol_padding_mask,
                                             )
        data_output = self.inner_model("fwd",
                                       output_times=output_times,
                                    embedder= self.no_inner_model.embedder,
                                    fused=  output_noinner["fused"],
                                    fused_mask=  output_noinner["fused_mask"])
        data_output["data_encoded"] = output_noinner["data_encoded"]
        data_output["data_embeded"] = output_noinner["data_embeded"]
        data_output["fused"] = output_noinner["fused"]
        return data_output
    def generate(self, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
        return self.fwd(data_input, input_times, output_times, symbol_input, symbol_padding_mask)

# class Combine_freeze_symbol(nn.Module):
#     def __init__(self, no_inner_model, inner_model):
#         super().__init__()
#         self.no_inner_model = no_inner_model
#         self.inner_model = inner_model
#
#     def forward(self, mode, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
#         if mode == "fwd":
#             return self.fwd(data_input, input_times, output_times, symbol_input, symbol_padding_mask)
#         elif mode == "generate":
#             return self.generate(data_input, input_times, output_times, symbol_input, symbol_padding_mask)
#         else:
#             raise Exception(f"Unknown mode: {mode}")
#
#     def fwd(self, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
#         output_noinner = self.no_inner_model("fwd",
#                                              symbol_input=symbol_input,
#                                              symbol_padding_mask=symbol_padding_mask)
#         data_output = self.inner_model("fwd",
#                                        data_input=data_input,
#                                        input_times=input_times,
#                                        output_times=output_times,
#                                        symbol_encoded=output_noinner["symbol_encoded"],
#                                        symbol_padding_mask=symbol_padding_mask, )
#         data_output["symbol_encoded"] = output_noinner["symbol_encoded"]
#         return data_output
#     def generate(self, data_input, input_times, output_times, symbol_input, symbol_padding_mask=None):
#         return self.fwd(data_input, input_times, output_times, symbol_input, symbol_padding_mask)

