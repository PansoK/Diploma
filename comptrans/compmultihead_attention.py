# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

from torch.autograd import Variable


@with_incremental_state
class MultiheadAttentionComp(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        cfg,
        cfg_comp,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cfg_comp = cfg_comp
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        #################
        # competitive format

        assert (cfg_comp.attention_competition_type == "softmax" or cfg_comp.attention_competition_type == "step")
        self.attention_competition_type = cfg_comp.attention_competition_type
        if self.attention_competition_type == "step":
            self.attention_heads_inactive = self.cfg_comp.attention_heads_inactive
            assert self.attention_heads_inactive < self.num_heads

        inference_activation_dropout_p = self.cfg_comp.attention_heads_inference_mlp_activ_dropout
        if inference_activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            inference_activation_dropout_p = cfg.relu_dropout or 0
        self.inference_activation_dropout_module = FairseqDropout(
            float(inference_activation_dropout_p), module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        # signature of each head
        self.using_weights_for_signatures = self.cfg_comp.attention_using_head_weigts
        if not cfg_comp.attention_using_head_weigts:
            self.head_signatures = nn.Parameter(torch.Tensor(
                self.num_heads,
                self.cfg_comp.attention_heads_signature_dim
                ))
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        # mlp that will produce inference vectors 
        self.inf_fc1 = self.build_fc1(
            self.embed_dim,
            self.cfg_comp.attention_heads_inference_mlp_hidden,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.inf_fc2 = self.build_fc2(
            self.cfg_comp.attention_heads_inference_mlp_hidden,
            self.cfg_comp.attention_heads_signature_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        # mlp that will produce signature vectors
        self.inf_sign_fc1 = self.build_fc1(
            3*self.head_dim*(1 + self.embed_dim),
            self.cfg_comp.attention_heads_inference_mlp_hidden,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.inf_sign_fc2 = self.build_fc2(
            self.cfg_comp.attention_heads_inference_mlp_hidden,
            self.cfg_comp.attention_heads_signature_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        '''   -> IF USED DON'T FORGET THE RESET PROCESS
        self.inf_sign_fc3 = self.build_fc2(
            self.cfg_comp.attention_heads_inference_mlp_hidden,
            self.cfg_comp.attention_heads_signature_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        '''
        #################

        self.reset_parameters(cfg)

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self, cfg):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.inf_fc1.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
            nn.init.xavier_uniform_(self.inf_fc2.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
            if self.attention_using_head_weigts:
                nn.init.xavier_uniform_(self.inf_sign_fc1.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
                nn.init.xavier_uniform_(self.inf_sign_fc2.weight, gain=nn.init.calculate_gain(cfg.activation_fn))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.inf_fc1.weight)
            nn.init.xavier_uniform_(self.inf_fc2.weight)
            if self.attention_using_head_weigts:
                nn.init.xavier_uniform_(self.inf_sign_fc1.weight)
                nn.init.xavier_uniform_(self.inf_sign_fc2.weight)

        if not self.cfg_comp.attention_using_head_weigts:
            nn.init.xavier_normal_(self.head_signatures)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def compute_signatures_from_weights(self):
        """
        This function uses the weights of the attention layer (q_proj, k_proj, v_proj) to compute the signature for each head

        q_proj: '(embed_dim, embed_dim) + bias if True'
        k_proj: '(self.kdim, embed_dim) + bias if True'
        v_proj: '(self.vdim, embed_dim) + bias if True'

        The weights that correspond to each head are proj.weight[i*head_dim:(i+1)*head_dim, :] and proj.bias[i*head_dim:(i+1)*head_dim]

        Returns:
            This returns the signature for each head in form '(num_heads, attention_heads_signature_dim)''
        """

        # splitting per head
        q_proj_weight = self.q_proj.weight.contiguous().view(self.num_heads, self.head_dim * self.embed_dim)
        q_proj_bias = self.q_proj.bias.contiguous().view(self.num_heads, self.head_dim)
        k_proj_weight = self.k_proj.weight.contiguous().view(self.num_heads, self.head_dim * self.embed_dim)
        k_proj_bias = self.k_proj.bias.contiguous().view(self.num_heads, self.head_dim)
        v_proj_weight = self.v_proj.weight.contiguous().view(self.num_heads, self.head_dim * self.embed_dim)
        v_proj_bias = self.v_proj.bias.contiguous().view(self.num_heads, self.head_dim)
        weights_per_head = torch.cat((q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias), dim=1)
        x = self.activation_fn(self.inf_sign_fc1(weights_per_head))
        x = self.inference_activation_dropout_module(x)
        x = self.inf_sign_fc2(x)
        #x = self.activation_fn(self.inf_sign_fc2(x))
        #x = self.inference_activation_dropout_module(x)
        #x = self.inf_sign_fc3(x)

        return x



    def inference_masking_step(self, x):
        """
        Args:
            x (Tensor): input to the ffns `(seq_len, batch, embed_dim)`

        Returns:
            attention heads mask equal to 1 at heads that will be dropped, of shape '(batch, number_of_heads, seq_len)'
        """

        t_size = (x.size(0), x.size(1), self.num_heads)
        res1 = Variable(torch.ones(t_size, device=x.get_device()))  # requires_grad=False maybe
        res2 = Variable(torch.zeros(t_size, device=x.get_device()))  # requires_grad=False maybe

        if self.attention_heads_inactive == 0:
            return res1.transpose(0, 2).transpose(1, 0)

        t = self.activation_fn(self.inf_fc1(x))
        t = self.inference_activation_dropout_module(t)
        t = self.inf_fc2(t)

        if self.cfg_comp.attention_using_head_weigts:
            head_signatures = self.compute_signatures_from_weights()
            t = torch.matmul(t, head_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_heads)
        else:
            t = torch.matmul(t, self.head_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_heads)
        #print(t.transpose(0, 2)[:, 0, :])

        topk, ind = t.topk(t.size()[2], dim=2)

        #t = t.scatter(2, ind[:, :, -self.attention_heads_inactive:], res1)  # filling inactive head positions with ones
        #c = t.scatter(2, ind[:, :, :-self.attention_heads_inactive], res2)  # filling active head positions with zeros
        #print(t[3, 15, :])
        t = t.scatter(2, ind[:, :, -self.attention_heads_inactive:], res2)  # filling inactive head positions with zeros
        #print(t[3, 15, :])
        c = t.scatter(2, ind[:, :, :-self.attention_heads_inactive], res1)  # filling active head positions with ones
        #print(c[3, 15, :])

        return c.transpose(0, 2).transpose(1, 0)

    def inference_masking_softmax(self, x):
        """
        Args:
            x (Tensor): input to the ffns `(seq_len, batch, embed_dim)`

        Returns:
            attention heads mask equal to 1 at heads that will be dropped, of shape '(batch, number_of_heads, seq_len)'
        """

        t = self.activation_fn(self.inf_fc1(x))
        t = self.inference_activation_dropout_module(t)
        t = self.inf_fc2(t)

        if self.cfg_comp.attention_using_head_weigts:
            head_signatures = self.compute_signatures_from_weights()
            t = torch.matmul(t, head_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_heads)
        else:
            t = torch.matmul(t, self.head_signatures.transpose(0, 1))  # of (seq_len, batch, n_of_heads)
        c = nn.functional.softmax(t, dim=2)

        return c.transpose(0, 2).transpose(1, 0)


    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        
        '''
        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        '''
        
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttentionComp._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # not None only for decoder self-attention; None for other cases
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask  # this is for teacher forcing; hiding is done by adding negative infinity

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        ############# comp
        with open("test_from_weights.txt",'a',encoding = 'utf-8') as f:
            #if self.attention_competition_type == "softmax":
            #    f.write(f"Before: {attn[100:104, 5, :16]}\n")

            attn = attn.view(
                bsz, self.num_heads, tgt_len, self.head_dim
            )

            #c = self.inference_masking(query)
            if self.attention_competition_type == "softmax":
                c = self.inference_masking_softmax(query)
            else:
                c = self.inference_masking_step(query)
            assert list(c.size()) == [bsz, self.num_heads, tgt_len]
            
            #if self.attention_competition_type == "softmax":
            #    print(torch.mean(c[:, :, :5], dim=0).data)
            if self.attention_competition_type == "softmax":
                f.write(f"Multplying: {c[0, :, 4].data}\n")
            #print(self.head_signatures[:, 10])
            if self.attention_competition_type == "softmax":
                if not self.cfg_comp.attention_using_head_weigts:
                    norm1 = torch.linalg.norm(self.head_signatures[0, :]).data
                    norm2 = torch.linalg.norm(self.head_signatures[1, :]).data
                    norm3 = torch.linalg.norm(self.head_signatures[2, :]).data
                    norm4 = torch.linalg.norm(self.head_signatures[3, :]).data
                    f.write(f"Head Norms: {norm1}, {norm2}, {norm3}, {norm4}\n")
                else:
                    signatures = self.compute_signatures_from_weights()
                    norm1 = torch.linalg.norm(signatures[0, :]).data
                    norm2 = torch.linalg.norm(signatures[1, :]).data
                    norm3 = torch.linalg.norm(signatures[2, :]).data
                    norm4 = torch.linalg.norm(signatures[3, :]).data
                    f.write(f"Head Norms: {norm1}, {norm2}, {norm3}, {norm4}\n")
            '''
            if self.attention_competition_type == "softmax":
                if not self.cfg_comp.attention_using_head_weigts:
                    cos1 = F.cosine_similarity(self.head_signatures[0, :], self.head_signatures[1, :], dim=0)
                    cos2 = F.cosine_similarity(self.head_signatures[0, :], self.head_signatures[2, :], dim=0)
                    cos3 = F.cosine_similarity(self.head_signatures[1, :], self.head_signatures[2, :], dim=0)
                    f.write(f"Sims: {cos1}, {cos2}, {cos3}\n")
                else:
                    signatures = self.compute_signatures_from_weights()
                    cos1 = F.cosine_similarity(signatures[0, :], signatures[1, :], dim=0)
                    cos2 = F.cosine_similarity(signatures[0, :], signatures[2, :], dim=0)
                    cos3 = F.cosine_similarity(signatures[1, :], signatures[2, :], dim=0)
                    f.write(f"Sims: {cos1}, {cos2}, {cos3}\n")
            '''
            if self.attention_competition_type == "softmax":
                cos1 = torch.mean(F.cosine_similarity(attn[:, 0, 4, :], attn[:, 1, 4, :], dim=1)).item()
                cos2 = torch.mean(F.cosine_similarity(attn[:, 0, 4, :], attn[:, 2, 4, :], dim=1)).item()
                cos3 = torch.mean(F.cosine_similarity(attn[:, 0, 4, :], attn[:, 3, 4, :], dim=1)).item()
                cos4 = torch.mean(F.cosine_similarity(attn[:, 1, 4, :], attn[:, 2, 4, :], dim=1)).item()
                cos5 = torch.mean(F.cosine_similarity(attn[:, 1, 4, :], attn[:, 3, 4, :], dim=1)).item()
                cos6 = torch.mean(F.cosine_similarity(attn[:, 2, 4, :], attn[:, 3, 4, :], dim=1)).item()
                f.write(f"Sims: {cos1}, {cos2}, {cos3}, {cos4}, {cos5}, {cos6}\n")


            #attn = attn.masked_fill(
            #    c.unsqueeze(3).to(torch.bool),
            #    1,  # changed it to one, so that even if this is used we do not have to change anything else
            #)
            attn = torch.mul(attn, c.unsqueeze(3))  # applying masking

            attn = attn.view(
                bsz*self.num_heads, tgt_len, self.head_dim
            )

            #if self.attention_competition_type == "softmax":
            #    f.write(f"After: {attn[100:104, 5, :16]}\n")
        #############

        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask.float(), filler.float()], dim=1
                )
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                new_key_padding_mask = torch.cat(
                    [filler.float(), key_padding_mask.float()], dim=1
                )
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
