# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm#, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
#from fairseq.models.transformer import (
#    TransformerConfig,
#)
from comptransformer_config import TransformerCompConfig
from compmultihead_attention import MultiheadAttentionComp
import datetime


class TransformerEncoderLayerCompBase(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, layer):
        super().__init__()
        self.cfg = cfg
        self.layer = layer
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg, name=("encoder_self", layer))
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        '''
        #######################
        # number of parallel ffns per encoder layer
        self.parallel_ffns = cfg.encoder.competitive.parallel_ffns
        inference_activation_dropout_p = cfg.encoder.competitive.parallel_ffns_inference_mlp_activ_dropout
        if inference_activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            inference_activation_dropout_p = cfg.relu_dropout or 0
        self.inference_activation_dropout_module = FairseqDropout(
            float(inference_activation_dropout_p), module_name=self.__class__.__name__
        )

        # https://github.com/pytorch/pytorch/issues/54147
        CHECK THIS OUT FIRST nn.Linear(embed_dim, output_dim, bias=False).weight (https://github.com/pytorch/fairseq/blob/00b6adfbdc58d473c9039a96c124e75f922e3808/fairseq/models/roberta/model.py#L366)
        self.fc1 = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            self.embed_dim+1,
            cfg.encoder.ffn_embed_dim
            ))
        self.fc2 = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            cfg.encoder.ffn_embed_dim+1,
            self.embed_dim
            ))

        # signature of each ffn
        self.ffn_signatures = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            cfg.encoder.competitive.parallel_ffns_signature_dim
            ))
        # mlp that will produce inference vectors 
        self.inf_fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.competitive.parallel_ffns_inference_mlp_hidden,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.inf_fc2 = self.build_fc2(
            cfg.encoder.competitive.parallel_ffns_inference_mlp_hidden,
            cfg.encoder.competitive.parallel_ffns_signature_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        '''

        #######################

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        #self.reset_parameters(cfg)

    def reset_parameters(self, cfg):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            for w in itertools.chain(self.fc1[:, :-1, :], self.fc2[:, :-1, :]):
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain(cfg.activation_fn))
            nn.init.xavier_uniform_(self.inf_fc1.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
            nn.init.xavier_uniform_(self.inf_fc2.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
        else:
            for w in itertools.chain(self.fc1[:, :-1, :], self.fc2[:, :-1, :]):
                nn.init.xavier_uniform_(w)
            nn.init.xavier_uniform_(self.inf_fc1.weight)
            nn.init.xavier_uniform_(self.inf_fc2.weight)

        # nn.init.xavier_uniform_(self.ffn_signatures)
        nn.init.xavier_normal_(self.ffn_signatures)  # run 1.1

        nn.init.xavier_normal_(self.fc1[:, -1, :])
        nn.init.xavier_normal_(self.fc2[:, -1, :])

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, cfg, name):
        #######################
        return MultiheadAttentionComp(
            name,
            cfg,
            cfg.encoder.competitive,
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        #######################
        '''
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        '''

    def residual_connection(self, x, residual):
        return residual + x

    def inference_matching(self, x):
        """
        Args:
            x (Tensor): input to the ffns `(seq_len, batch, embed_dim)`

        Returns:
            softmax results for each combination of element x_i and possible fnn of shape '(number_of_functions, seq_len, batch)'
        """
        t = self.activation_fn(self.inf_fc1(x))
        t = self.inference_activation_dropout_module(t)
        t = self.inf_fc2(t)

        t = torch.matmul(t, self.ffn_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_functions)
        c = nn.functional.softmax(t, dim=2)
        return c.transpose(0, 2).transpose(1, 2)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        print_data_for_seq: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
            print_data_for_seq (Tensor, optional): an [N, 2] Tensor with information
                about the sequences for which we print metrics. It contains the sequence 
                identifiers in the first column and the sequence position in the batch
                in the second one.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        # just to make sure that everything is correct
        seq_len, bsz, _ = x.size()

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            print_data_for_seq=print_data_for_seq,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        #######################
        '''
        # starting with the inference part
        c = self.inference_matching(x)  # of (number_of_functions, seq_len, batch)
        assert list(c.size()) == [self.parallel_ffns, seq_len, bsz]
        # adding ones for biases
        x = torch.cat((x, torch.ones((x.size(0), x.size(1), 1), dtype=x.dtype, device=x.get_device())), dim=2)
        # applying linear layer
        x = torch.matmul(x.unsqueeze(1), self.fc1).transpose(1, 2)  # of (seq_len, batch, number_of_functions, ffn_embed_dim)
        assert list(x.size()) == [seq_len, bsz, self.parallel_ffns, self.cfg.encoder.ffn_embed_dim]
        x = self.activation_fn(x)
        # adding ones for biases
        x = torch.cat((x, torch.ones((x.size(0), x.size(1), x.size(2), 1), dtype=x.dtype, device=x.get_device())), dim=3)
        x = self.activation_dropout_module(x)
        #print(x.size())
        #print(bsz, seq_len, self.parallel_ffns, self.embed_dim, self.cfg.encoder.ffn_embed_dim)
        # print(c[:, 2, 0])
        #cos1 = F.cosine_similarity(self.ffn_signatures[0, :], self.ffn_signatures[1, :], dim=0)
        #cos2 = F.cosine_similarity(self.ffn_signatures[0, :], self.ffn_signatures[2, :], dim=0)
        #cos3 = F.cosine_similarity(self.ffn_signatures[1, :], self.ffn_signatures[2, :], dim=0)
        #print(cos1, cos2, cos3)

        x = (
            x.contiguous()  # check https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do
            .view(bsz * seq_len, self.parallel_ffns, self.cfg.encoder.ffn_embed_dim+1)  # returns a new tensor with the same data as the self tensor but of a different shape
            # essentially here we are breaking it into heads, considering every head as a special case
            .transpose(0, 1)
        )
        x = torch.bmm(x, self.fc2)  # of (number_of_functions, seq_len * batch, self.embed_dim)
        assert list(x.size()) == [self.parallel_ffns, seq_len * bsz, self.embed_dim]
        x = x.view(self.parallel_ffns, seq_len, bsz, self.embed_dim)

        #cos1 = torch.mean(F.cosine_similarity(x[0, 3, :, :], x[1, 3, :, :], dim=1))
        #cos2 = torch.mean(F.cosine_similarity(x[0, 3, :, :], x[2, 3, :, :], dim=1))
        #cos3 = torch.mean(F.cosine_similarity(x[1, 3, :, :], x[2, 3, :, :], dim=1))
        #print(cos1, cos2, cos3)

        x = torch.mul(x, c.unsqueeze(3))  # broadcasting happens, shape not is (number_of_functions, seq_len, batch, self.embed_dim)
        x = torch.sum(x, dim=0)
        '''
        #######################
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        '''
        with open("/home/panso014/diploma/code/train_1_1_4so_4wt_4wt_iter_3_to_print.txt",'a', encoding = 'utf-8') as f:
            data_dict = {"name": ("encoder_layer", self.layer)}
            now = datetime.datetime.now()
            data_dict["time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            data_dict["fc1"] = [torch.linalg.norm(self.fc1.weight).item(), torch.linalg.norm(self.fc1.bias).item()]
            f.write(str(data_dict) + "\n")
        '''
        #######################

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


# backward compatible with the legacy argparse format
class TransformerEncoderLayerComp(TransformerEncoderLayerCompBase):
    def __init__(self, args, layer):
        super().__init__(TransformerCompConfig.from_namespace(args), layer)
        self.args = args

    def build_self_attention(self, embed_dim, args, name):
        return super().build_self_attention(
            embed_dim, TransformerCompConfig.from_namespace(args), name
        )


class TransformerDecoderLayerCompBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, layer, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            name=("decoder_self", layer),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = (
            LayerNorm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
            if scale_heads
            else None
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg, name=("encoder_decoder", layer))
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.ffn_layernorm = (
            LayerNorm(cfg.decoder.ffn_embed_dim)
            if utils.safe_getattr(cfg, "scale_fc", False)
            else None
        )
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if utils.safe_getattr(cfg, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        #######################
        '''
        self.parallel_ffns = cfg.decoder.competitive.parallel_ffns
        inference_activation_dropout_p = cfg.decoder.competitive.parallel_ffns_inference_mlp_activ_dropout
        if inference_activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            inference_activation_dropout_p = cfg.relu_dropout or 0
        self.inference_activation_dropout_module = FairseqDropout(
            float(inference_activation_dropout_p), module_name=self.__class__.__name__
        )

        # https://github.com/pytorch/pytorch/issues/54147
        self.fc1 = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            self.embed_dim+1,
            cfg.decoder.ffn_embed_dim
            ))
        self.fc2 = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            cfg.decoder.ffn_embed_dim+1,
            self.embed_dim
            ))

        # signature of each ffn
        self.ffn_signatures = nn.Parameter(torch.Tensor(
            self.parallel_ffns,
            cfg.decoder.competitive.parallel_ffns_signature_dim
            ))
        # mlp that will produce inference vectors 
        self.inf_fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.competitive.parallel_ffns_inference_mlp_hidden,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.inf_fc2 = self.build_fc2(
            cfg.decoder.competitive.parallel_ffns_inference_mlp_hidden,
            cfg.decoder.competitive.parallel_ffns_signature_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        '''
        #######################

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

        #self.reset_parameters(cfg)

    def reset_parameters(self, cfg):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            for w in itertools.chain(self.fc1[:, :-1, :], self.fc2[:, :-1, :]):
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain(cfg.activation_fn))
            nn.init.xavier_uniform_(self.inf_fc1.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
            nn.init.xavier_uniform_(self.inf_fc2.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
        else:
            for w in itertools.chain(self.fc1[:, :-1, :], self.fc2[:, :-1, :]):
                nn.init.xavier_uniform_(w)
            nn.init.xavier_uniform_(self.inf_fc1.weight)
            nn.init.xavier_uniform_(self.inf_fc2.weight)

        # nn.init.xavier_uniform_(self.ffn_signatures)
        nn.init.xavier_normal_(self.ffn_signatures)  # run 1.1

        nn.init.xavier_normal_(self.fc1[:, -1, :])
        nn.init.xavier_normal_(self.fc2[:, -1, :])

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    #######################
    def build_self_attention(
        self, embed_dim, cfg, name, add_bias_kv=False, add_zero_attn=False
        ):
        return MultiheadAttentionComp(
            name,
            cfg,
            cfg.decoder.competitive,
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        '''
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        '''

    def build_encoder_attention(self, embed_dim, cfg, name):
        return MultiheadAttentionComp(
            name,
            cfg,
            cfg.decoder.competitivecrossattn,
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        '''
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        '''

    #######################

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def inference_matching(self, x):
        """
        Args:
            x (Tensor): input to the ffns `(seq_len, batch, embed_dim)`

        Returns:
            softmax results for each combination of element x_i and possible fnn of shape '(number_of_functions, seq_len, batch)'
        """
        t = self.activation_fn(self.inf_fc1(x))
        t = self.inference_activation_dropout_module(t)
        t = self.inf_fc2(t)

        t = torch.matmul(t, self.ffn_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_functions)
        c = nn.functional.softmax(t, dim=2)
        return c.transpose(0, 2).transpose(1, 2)


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        print_data_for_seq: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
            print_data_for_seq (Tensor, optional): an [N, 2] Tensor with information
                about the sequences for which we print metrics. It contains the sequence 
                identifiers in the first column and the sequence position in the batch
                in the second one.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        seq_len, bsz, _ = x.size()

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,  # this comes from the queries but its the same in self-attention and not used in cross-attention
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            print_data_for_seq=print_data_for_seq,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                print_data_for_seq=print_data_for_seq,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        #######################
        '''
        # starting with the inference part
        c = self.inference_matching(x)  # of (number_of_functions, seq_len, batch)
        assert list(c.size()) == [self.parallel_ffns, seq_len, bsz]
        # adding ones for biases
        x = torch.cat((x, torch.ones((x.size(0), x.size(1), 1), dtype=x.dtype, device=x.get_device())), dim=2)
        # applying linear layer
        x = torch.matmul(x.unsqueeze(1), self.fc1).transpose(1, 2)  # of (seq_len, batch, number_of_functions, ffn_embed_dim)
        assert list(x.size()) == [seq_len, bsz, self.parallel_ffns, self.cfg.decoder.ffn_embed_dim]
        x = self.activation_fn(x)
        # adding ones for biases
        x = torch.cat((x, torch.ones((x.size(0), x.size(1), x.size(2), 1), dtype=x.dtype, device=x.get_device())), dim=3)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        #print(x.size())
        #print(bsz, seq_len, self.parallel_ffns, self.embed_dim, self.cfg.encoder.ffn_embed_dim)
        # print(c[:, 2, 0])
        #cos1 = F.cosine_similarity(self.ffn_signatures[0, :], self.ffn_signatures[1, :], dim=0)
        #cos2 = F.cosine_similarity(self.ffn_signatures[0, :], self.ffn_signatures[2, :], dim=0)
        #cos3 = F.cosine_similarity(self.ffn_signatures[1, :], self.ffn_signatures[2, :], dim=0)
        #print(cos1, cos2, cos3)

        x = (
            x.contiguous()  # check https://stackoverflow.com/questions/48915810/pytorch-what-does-contiguous-do
            .view(bsz * seq_len, self.parallel_ffns, self.cfg.encoder.ffn_embed_dim+1)  # returns a new tensor with the same data as the self tensor but of a different shape
            # essentially here we are breaking it into heads, considering every head as a special case
            .transpose(0, 1)
        )
        x = torch.bmm(x, self.fc2)  # of (number_of_functions, seq_len * batch, self.embed_dim)
        assert list(x.size()) == [self.parallel_ffns, seq_len * bsz, self.embed_dim]
        x = x.view(self.parallel_ffns, seq_len, bsz, self.embed_dim)

        #cos1 = torch.mean(F.cosine_similarity(x[0, 3, :, :], x[1, 3, :, :], dim=1))
        #cos2 = torch.mean(F.cosine_similarity(x[0, 3, :, :], x[2, 3, :, :], dim=1))
        #cos3 = torch.mean(F.cosine_similarity(x[1, 3, :, :], x[2, 3, :, :], dim=1))
        #print(cos1, cos2, cos3)

        x = torch.mul(x, c.unsqueeze(3))  # broadcasting happens, shape not is (number_of_functions, seq_len, batch, self.embed_dim)
        x = torch.sum(x, dim=0)
        '''
        #######################

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        #######################

        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayerComp(TransformerDecoderLayerCompBase):
    def __init__(
        self, args, layer, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerCompConfig.from_namespace(args),
            layer,
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, name, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerCompConfig.from_namespace(args),
            name,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args, name):
        return super().build_encoder_attention(
            embed_dim,
            TransformerCompConfig.from_namespace(args),
            name
        )
