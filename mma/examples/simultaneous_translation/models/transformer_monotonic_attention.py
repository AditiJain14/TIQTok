# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, NamedTuple, Optional

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.simultaneous_translation.modules.monotonic_transformer_layer import (
    TransformerMonotonicDecoderLayer,
    TransformerMonotonicEncoderLayer,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
)
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

TransformerMonotonicDecoderOut = NamedTuple(
    "TransformerMonotonicDecoderOut",
    [
        ("action", int),
        ("p_choose", Optional[Tensor]),
        ("attn_list", Optional[List[Optional[Dict[str, Tensor]]]]),
        ("step_list", Optional[List[Optional[Tensor]]]),
        ("encoder_out", Optional[Dict[str, List[Tensor]]]),
        ("encoder_padding_mask", Optional[Tensor]),
    ],
)


@register_model("transformer_unidirectional")
class TransformerUnidirectionalModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)


@register_model("transformer_monotonic")
class TransformerModelSimulTrans(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerMonotonicEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerMonotonicDecoder(args, tgt_dict, embed_tokens)

    def _indices_from_states(self, states):
        if type(states["indices"]["src"]) == list:
            if next(self.parameters()).is_cuda:
                tensor = torch.cuda.LongTensor
            else:
                tensor = torch.LongTensor

            src_indices = tensor(
                [states["indices"]["src"][: 1 + states["steps"]["src"]]]
            )

            tgt_indices = tensor(
                [[self.decoder.dictionary.eos()] + states["indices"]["tgt"]]
            )
        else:
            src_indices = states["indices"]["src"][: 1 + states["steps"]["src"]]
            tgt_indices = states["indices"]["tgt"]

        return src_indices, None, tgt_indices


class TransformerMonotonicEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerMonotonicEncoderLayer(args) for i in range(args.encoder_layers)]
        )


class TransformerMonotonicDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerMonotonicDecoderLayer(
                    args, no_encoder_attn, need_alpha=(0 == i)
                )
                for i in range(args.decoder_layers)
            ]
        )

    def pre_attention(
        self,
        prev_output_tokens,
        encoder_out_dict: Dict[str, List[Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        positions = (
            self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = encoder_out_dict["encoder_out"][0]

        if "encoder_padding_mask" in encoder_out_dict:
            encoder_padding_mask = (
                encoder_out_dict["encoder_padding_mask"][0]
                if encoder_out_dict["encoder_padding_mask"]
                and len(encoder_out_dict["encoder_padding_mask"]) > 0
                else None
            )
        else:
            encoder_padding_mask = None

        return x, encoder_out, encoder_padding_mask

    def post_attention(self, x):
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x

    def clear_cache(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        end_id: Optional[int] = None,
    ):
        """
        Clear cache in the monotonic layers.
        The cache is generated because of a forward pass of decode but no prediction.
        end_id is the last idx of the layers
        """
        if end_id is None:
            end_id = len(self.layers)

        for index, layer in enumerate(self.layers):
            if index < end_id:
                layer.prune_incremental_state(incremental_state)

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,  # unused
        alignment_layer: Optional[int] = None,  # unused
        alignment_heads: Optional[int] = None,  # unsed
        pre_alpha=None,
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # incremental_state = None
        assert encoder_out is not None
        (x, encoder_outs, encoder_padding_mask) = self.pre_attention(
            prev_output_tokens, encoder_out, incremental_state
        )
        attn = None
        inner_states = [x]
        attn_list: List[Optional[Dict[str, Tensor]]] = []
        step_list: List[Optional[Tensor]] = []

        p_choose = torch.tensor([1.0])

        betas = []

        for i, layer in enumerate(self.layers):

            x, attn, _ = layer(
                x=x,
                encoder_out=encoder_outs,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x)
                if incremental_state is None
                else None,
                pre_alpha=pre_alpha,
            )
            pre_alpha = attn
            betas.append(attn["beta"])

            inner_states.append(x)
            attn_list.append(attn)

            if incremental_state is not None:
                curr_steps = layer.get_head_steps(incremental_state)
                step_list.append(curr_steps)
                if_online = incremental_state["online"]["only"]
                assert if_online is not None
                if if_online.to(torch.bool):
                    # Online indicates that the encoder states are still changing
                    assert attn is not None
                    assert curr_steps is not None
                    p_choose = (
                        attn["p_choose"].squeeze(0).squeeze(1).gather(1, curr_steps.t())
                    )

                    new_steps = curr_steps + (p_choose < 0.5).t().type_as(curr_steps)
                    src = incremental_state["steps"]["src"]
                    assert src is not None

                    if (new_steps >= src).any():
                        # We need to prune the last self_attn saved_state
                        # if model decide not to read
                        # otherwise there will be duplicated saved_state
                        self.clear_cache(incremental_state, i + 1)

                        return x, TransformerMonotonicDecoderOut(
                            action=0,
                            p_choose=p_choose,
                            attn_list=None,
                            step_list=None,
                            encoder_out=None,
                            encoder_padding_mask=None,
                        )

        x = self.post_attention(x)

        pre_alpha["beta"] = torch.mean(torch.stack(betas), 0)

        return x, {
            "action": 1,
            "p_choose": p_choose,
            #'attn_list':attn_list,
            "attn_list": [pre_alpha],
            "step_list": step_list,
            "encoder_out": encoder_out,
            "encoder_padding_mask": encoder_padding_mask,
        }

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if "fastest_step" in incremental_state:
            incremental_state["fastest_step"] = incremental_state[
                "fastest_step"
            ].index_select(0, new_order)


@register_model_architecture("transformer_monotonic", "transformer_monotonic")
def base_monotonic_architecture(args):
    base_architecture(args)
    args.encoder_unidirectional = getattr(args, "encoder_unidirectional", False)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_iwslt_de_en"
)
def transformer_monotonic_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
    base_monotonic_architecture(args)

    
@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_wmt"
)
def transformer_monotonic_wmt(args):
    base_monotonic_architecture(args)
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



# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_de_big"
)
def transformer_monotonic_vaswani_wmt_en_de_big(args):
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture(
    "transformer_monotonic", "transformer_monotonic_vaswani_wmt_en_fr_big"
)
def transformer_monotonic_vaswani_wmt_en_fr_big(args):
    transformer_monotonic_vaswani_wmt_en_fr_big(args)


@register_model_architecture(
    "transformer_unidirectional", "transformer_unidirectional_iwslt_de_en"
)
def transformer_unidirectional_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
