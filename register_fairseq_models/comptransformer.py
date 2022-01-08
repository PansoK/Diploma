from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    #TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
#from fairseq.models.transformer.transformer_base import (
#    TransformerModelBase,
#)

import sys
sys.path.insert(1, '/home/panso014/diploma/code/packages/src/comptrans')
from transformer_base import TransformerCompModelBase
from comptransformer_config import TransformerCompConfig

@register_model("comptransformer_test")
class TransformerCompModel(TransformerCompModelBase):
    """
    This is the legacy implementation of the transformer model that
    uses argparse for configuration.
    """

    def __init__(self, args, encoder, decoder):
        cfg = TransformerCompConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        # do not set defaults so that settings defaults from various architectures still works
        gen_parser_from_dataclass(
            parser, TransformerCompConfig(), delete_default=True, with_prefix=""
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            args.share_decoder_input_output_embed = True

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        if not args.share_all_embeddings:
            args.min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
        cfg = TransformerCompConfig.from_namespace(args)
        return super().build_model(cfg, task)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            TransformerCompConfig.from_namespace(args), dictionary, embed_dim, path
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            TransformerCompConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            TransformerCompConfig.from_namespace(args), tgt_dict, embed_tokens
        )


# architectures

@register_model_architecture("comptransformer_test", "comptransformer_test")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # for the competitive shit
    args.parallel_ffns_inference_training_type = getattr(args, "parallel_ffns_inference_training_type", "together")
    args.parallel_ffns_inference_training_type_sequential = getattr(args, "parallel_ffns_inference_training_type_sequential", 0.0)
    args.attention_heads_inference_training_type = getattr(args, "attention_heads_inference_training_type", "together")
    args.attention_heads_inference_training_type_sequential = getattr(args, "attention_heads_inference_training_type_sequential", 0.0)
    args.encoder_competitive_parallel_ffns = getattr(args, "encoder_competitive_parallel_ffns", 1)
    args.encoder_competitive_parallel_ffns_signature_dim = getattr(args, "encoder_competitive_parallel_ffns_signature_dim", 24)
    args.encoder_competitive_parallel_ffns_inference_mlp_hidden = getattr(args, "encoder_competitive_parallel_ffns_inference_mlp_hidden", 128)
    args.encoder_competitive_parallel_ffns_inference_mlp_activ_dropout = getattr(args, "encoder_competitive_parallel_ffns_inference_mlp_activ_dropout", 0.0)
    args.encoder_competitive_attention_competition_type = getattr(args, "encoder_competitive_attention_competition_type", "step")
    args.encoder_competitive_attention_heads_inactive = getattr(args, "encoder_competitive_attention_heads_inactive", 0)
    args.encoder_competitive_attention_heads_signature_dim= getattr(args, "encoder_competitive_attention_heads_signature_dim", 24)
    args.encoder_competitive_attention_heads_inference_mlp_hidden = getattr(args, "encoder_competitive_attention_heads_inference_mlp_hidden", 128)
    args.encoder_competitive_attention_using_head_weigts = getattr(args, "encoder_competitive_attention_using_head_weigts", False)
    args.encoder_competitive_attention_heads_inference_mlp_activ_dropout = getattr(args, "encoder_competitive_attention_heads_inference_mlp_activ_dropout", 0.0)
    args.decoder_competitive_parallel_ffns = getattr(args, "decoder_competitive_parallel_ffns", 1)
    args.decoder_competitive_parallel_ffns_signature_dim = getattr(args, "decoder_competitive_parallel_ffns_signature_dim", 24)
    args.decoder_competitive_parallel_ffns_inference_mlp_hidden = getattr(args, "decoder_competitive_parallel_ffns_inference_mlp_hidden", 128)
    args.decoder_competitive_attention_using_head_weigts = getattr(args, "decoder_competitive_attention_using_head_weigts", False)
    args.decoder_competitive_parallel_ffns_inference_mlp_activ_dropout = getattr(args, "decoder_competitive_parallel_ffns_inference_mlp_activ_dropout", 0.0)
    args.decoder_competitive_attention_competition_type = getattr(args, "decoder_competitive_attention_competition_type", "step")
    args.decoder_competitive_attention_heads_inactive = getattr(args, "decoder_competitive_attention_heads_inactive", 0)
    args.decoder_competitive_attention_heads_signature_dim= getattr(args, "decoder_competitive_attention_heads_signature_dim", 24)
    args.decoder_competitive_attention_heads_inference_mlp_hidden = getattr(args, "decoder_competitive_attention_heads_inference_mlp_hidden", 128)
    args.decoder_competitive_attention_heads_inference_mlp_activ_dropout = getattr(args, "decoder_competitive_attention_heads_inference_mlp_activ_dropout", 0.0)
    args.decoder_competitivecrossattn_attention_competition_type = getattr(args, "decoder_competitivecrossattn_attention_competition_type", "step")
    args.decoder_competitivecrossattn_attention_heads_inactive = getattr(args, "decoder_competitivecrossattn_attention_heads_inactive", 0)
    args.decoder_competitivecrossattn_attention_heads_signature_dim= getattr(args, "decoder_competitivecrossattn_attention_heads_signature_dim", 24)
    args.decoder_competitivecrossattn_attention_heads_inference_mlp_hidden = getattr(args, "decoder_competitivecrossattn_attention_heads_inference_mlp_hidden", 128)
    args.decoder_competitivecrossattn_attention_using_head_weigts = getattr(args, "decoder_competitivecrossattn_attention_using_head_weigts", False)
    args.decoder_competitivecrossattn_attention_heads_inference_mlp_activ_dropout = getattr(args, "decoder_competitivecrossattn_attention_heads_inference_mlp_activ_dropout", 0.0)

@register_model_architecture("comptransformer_test", "comptransformer_test_nmt_tiny")
def transformer_nmt_tiny(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.max_source_positions = getattr(args, "max_source_positions", 128)
    args.max_target_positions = getattr(args, "max_target_positions", 128)
    base_architecture(args)

@register_model_architecture("comptransformer_test", "comptransformer_test_nmt_mini")
def transformer_nmt_mini(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.max_source_positions = getattr(args, "max_source_positions", 128)
    args.max_target_positions = getattr(args, "max_target_positions", 128)
    base_architecture(args)

@register_model_architecture("comptransformer_test", "comptransformer_test_nmt_small_to_mid")
def transformer_nmt_small_to_mid(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)