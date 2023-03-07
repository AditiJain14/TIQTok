# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

from examples.simultaneous_translation.utils.latency import (
    LatencyTraining
)
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)

from fairseq.criterions.label_smoothed_cross_entropy_latency_augmented import (
    LatencyAugmentedLabelSmoothedCrossEntropyCriterion
)

#logger = logging.getLogger(__name__)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # import ipdb; ipdb.set_trace()
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    #import ipdb; ipdb.set_trace()
    return loss, nll_loss


@register_criterion('latency_augmented_label_smoothed_cross_entropy_cbmi')
class LatencyAugmentedLabelSmoothedCrossEntropyCriterionCBMI(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, 
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        latency_weight_avg,
        latency_weight_avg_type,
        latency_weight_var,
        latency_weight_var_type,
        mass_preservation,
        average_method,
        dual_weight,
        train_only_lm,
        lm_label_smoothing=0.1,
        token_scale=0.0,
        sentence_scale=0.0,
        pretrain_steps=100000,
        without_latency_steps=0,
        lm_rate=0.01,
        finetune_fix_lm=False,):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            dual_weight,
        )
        self.eps = label_smoothing
        self.lm_eps = lm_label_smoothing
        self.latency_weight_avg = latency_weight_avg
        self.latency_weight_avg_type = latency_weight_avg_type
        self.latency_weight_var = latency_weight_var
        self.latency_weight_var_type = latency_weight_var_type
        self.mass_preservation = mass_preservation
        self.average_method = average_method
        self.token_scale = token_scale
        self.sentence_scale = sentence_scale
        self.pretrain_steps = pretrain_steps
        self.without_latency_steps = without_latency_steps
        self.lm_rate = lm_rate
        self.num_updates=-1
        self.finetune_fix_lm=finetune_fix_lm
        self.train_only_lm=train_only_lm
        self.latency_train = LatencyTraining(
            self.latency_weight_avg,
            self.latency_weight_var,
            self.latency_weight_avg_type,
            self.latency_weight_var_type,
            self.mass_preservation,
            self.average_method,
        )

    @staticmethod
    def add_args(parser):
        super(
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion,
            LatencyAugmentedLabelSmoothedCrossEntropyCriterion,
        ).add_args(parser)

        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument("--latency-weight-avg", default=0., type=float, metavar='D',
                            help="Average loss weight")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="Variance loss weight")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="Statistics for Average loss type")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="Statistics for variance loss type")
        parser.add_argument("--average-method", default="weighted_average",
                            help="Average loss type")
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on
        #args for CBMI
        parser.add_argument('--lm-label-smoothing', default=0.1, type=float, 
                            help='epsilon for language model label smoothing, 0 means no label smoothing')
        parser.add_argument('--token-scale', default=0.0, type=float, 
                            help='hyperparameter for token cbmi')
        parser.add_argument('--sentence-scale', default=0.0, type=float, 
                            help='hyperparameter for sentence cbmi')
        parser.add_argument('--pretrain-steps', default=10000, type=int, 
                            help='step for ending pretrain and starting finetune')
        parser.add_argument('--without-latency-steps', default=0, type=int, 
                            help='step for training without latency loss')
        parser.add_argument('--lm-rate', default=0.01, type=float, 
                            help='lm loss rate')
        parser.add_argument('--finetune-fix-lm', default=False, type=bool, 
                            help='fix language model when finetuning')
        parser.add_argument('--train-only-lm', action='store_true', 
                            help='Set to train only LM')

    def forward(self, model, sample, reduce=True, num_updates=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        dual_path = True if self.dual_weight == 1.0 else False

        # if dual_path is False, dual_loss will be None
        # TODO: handle this to avoid malicious behaviour
        net_output, back_net_output, dual_loss, back_data, lm_net_output = model(**sample["net_input"], dual=dual_path, lm_out=self.training)

        # get forward model loss
        # lm_loss = torch.Tensor([0.0])
        if self.training:
            self.num_updates = num_updates
            assert self.num_updates > -1, "num_updates is not being updated"

            loss, nll_loss, lm_loss, lm_nll_loss, log = self.compute_loss(model, net_output, lm_net_output, sample, reduce=reduce)

            
            if self.train_only_lm:
                loss=lm_loss

            else:
                if self.finetune_fix_lm or self.num_updates > self.pretrain_steps:
                    lm_loss = lm_loss.detach()

                if self.num_updates < self.pretrain_steps:
                    loss = loss + self.lm_rate*10 * lm_loss
                else:
                #     if(self.num_updates)>3000:
                    loss = loss + self.lm_rate * lm_loss
        else:
                
            loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)
            

        # compute backward only if dual_path is set to true
        # TODO: make the changes backward compatible
        if dual_path:
            # dual_loss is None when dual=False
            assert dual_loss is not None, "dual_loss is None. Check model's forward method when dual=True."

            # prepare backward sample
            back_sample = {}
            back_sample["id"] = sample["id"].contiguous()
            back_sample["nsentences"] = sample["nsentences"]
            back_sample["ntokens"] = (sample["net_input"]["src_tokens"] > 1).sum()
            back_sample["net_input"] = back_data
            back_sample["target"] = sample["net_input"]["src_tokens"].contiguous()

            # get backward model loss
            back_loss, back_nll_loss = self.compute_loss(
                model, back_net_output, back_sample, reduce=reduce
            )

            loss = loss + back_loss + self.dual_weight * dual_loss

        
        nll_loss = nll_loss #+ back_nll_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        num_tokens = sample["ntokens"]
        num_sentences = sample["target"].size(0)

        if dual_path:
            back_sample_size = (
                back_sample["target"].size(0)
                if self.sentence_avg
                else back_sample["ntokens"]
            )

            sample_size = sample_size + back_sample_size
            num_tokens = num_tokens + back_sample["ntokens"]
            num_sentences = num_sentences + back_sample["target"].size(0)

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "lm_loss": lm_loss.data if self.training else 0.0,
            "lm_nll_loss": lm_nll_loss.data if self.training else 0.0,
            "ntokens": num_tokens,
            "nsentences": num_sentences,
            "sample_size": sample_size,
        }

        # validation does not return additionl log dict
        if self.training:
            for key in log:
                logging_output[key] = log[key]

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output
    

    def vanilla_compute_loss(self, model, net_output, lm_output ,sample, reduce=True):
        # lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        
        nmt_logits = net_output[0]
        lm_logits = lm_output[0]
        nmt_probs = utils.softmax(nmt_logits, -1).reshape(-1, nmt_logits.shape[-1])
        lm_probs = utils.softmax(lm_logits, -1).reshape(-1, lm_logits.shape[-1])
        nmt_lprobs = torch.log(nmt_probs)
        lm_lprobs = torch.log(lm_probs)
        
        target = sample["target"]
        pad_mask = target.ne(self.padding_idx)
        shape = target.shape
        target = target.reshape(-1)
        if target.dim() == lm_logits.dim() - 1:
            target = target.unsqueeze(-1)
        
        nmt_loss, nmt_nll_loss = label_smoothed_nll_loss(
            nmt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )

        lm_loss, lm_nll_loss = label_smoothed_nll_loss(
            lm_lprobs,
            target,
            self.lm_eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        num_updates = self.num_updates
        if num_updates > self.pretrain_steps and not self.train_only_lm:
            cbmi = torch.log(nmt_probs / (lm_probs + 1e-9))    # in case that lm_probs are too little
            cbmi = cbmi.detach()
            golden_cbmi = torch.gather(cbmi, -1, index=target.unsqueeze(-1))
            # token-weight
            token_cbmi = golden_cbmi.reshape(shape)
            mean_token_cbmi = (token_cbmi * pad_mask).sum(-1, keepdims=True) / pad_mask.sum(-1, keepdims=True)
            std_token_cbmi = torch.sqrt(torch.sum((token_cbmi - mean_token_cbmi) ** 2 * pad_mask, -1, keepdims=True) / pad_mask.shape[-1])
            norm_token_cbmi = (token_cbmi - mean_token_cbmi) / std_token_cbmi
            token_weight = torch.where(self.token_scale * norm_token_cbmi + 1.0 >= 0, 
                                       self.token_scale * norm_token_cbmi + 1.0, 
                                       torch.zeros_like(norm_token_cbmi))
            # sentence-weight
            sentence_cbmi = mean_token_cbmi
            mean_sentence_cbmi = sentence_cbmi.mean(0, keepdims=True)
            std_sentence_cbmi = torch.sqrt(torch.sum((sentence_cbmi - mean_sentence_cbmi) ** 2, 0, keepdims=True) / pad_mask.shape[-1])
            norm_sentence_cbmi = (sentence_cbmi - mean_sentence_cbmi) / std_sentence_cbmi
            sentence_weight = torch.where(self.sentence_scale * norm_sentence_cbmi + 1.0 >= 0, 
                                          self.sentence_scale * norm_sentence_cbmi + 1.0, 
                                          torch.zeros_like(norm_sentence_cbmi))
            # final-weight
            weight = token_weight * sentence_weight
            weight = weight.detach()
            nmt_loss = nmt_loss.reshape(shape)
            nmt_loss = weight * nmt_loss 
            # logging output
            mean_cbmi = (token_cbmi * pad_mask).sum() / pad_mask.sum()
            std_cbmi = torch.sqrt(((token_cbmi - mean_cbmi) ** 2 * pad_mask).sum() / pad_mask.sum())
            max_weight = weight.max()
            min_weight = weight.min()
            zero_rate = torch.div((weight.eq(0) * pad_mask).sum(), pad_mask.sum())
        else:
            mean_cbmi = 0.0
            std_cbmi = 0.0
            max_weight = 0.0
            min_weight = 0.0
            zero_rate = 0.0
            
        logging_output = {
            "mean_cbmi": mean_cbmi, 
            "std_cbmi": std_cbmi, 
            "max_weight": max_weight, 
            "min_weight": min_weight, 
            "zero_rate": zero_rate,
        }
        if reduce:
            nmt_loss = nmt_loss.sum()
            nmt_nll_loss = nmt_nll_loss.sum()

        return nmt_loss, nmt_nll_loss, lm_loss, lm_nll_loss, logging_output

    def compute_loss(self, model, net_output, lm_output, sample, reduce=True):
        # Compute cross entropy loss first
        #loss, nll_loss = self.vanilla_compute_loss(model, net_output, sample, reduce)
        loss, nll_loss, lm_loss, lm_nll_loss, logging_output = self.vanilla_compute_loss(model, net_output, lm_output ,sample, reduce)
        # Obtain the expected alignment
        if self.num_updates>self.without_latency_steps:
            attn_list = [item["alpha"] for item in net_output[-1]["attn_list"]]

            target_padding_mask = model.get_targets(sample, net_output).eq(self.padding_idx)

            source_padding_mask = net_output[-1].get("encoder_padding_mask", None)

            # Get latency loss
            latency_loss = self.latency_train.loss(
                attn_list, source_padding_mask, target_padding_mask)

            loss += latency_loss

        return loss, nll_loss, lm_loss, lm_nll_loss, logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get("lm_loss", 0) for log in logging_outputs)
        lm_nll_loss_sum = sum(log.get("lm_nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mean_cbmi_sum = sum(log.get("mean_cbmi", 0) for log in logging_outputs)
        std_cbmi_sum = sum(log.get("std_cbmi", 0) for log in logging_outputs)
        max_weight_sum = sum(log.get("max_weight", 0) for log in logging_outputs)
        min_weight_sum = sum(log.get("min_weight", 0) for log in logging_outputs)
        zero_rate_sum = sum(log.get("zero_rate", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        if not float(lm_loss_sum)==float(0):
            metrics.log_scalar(
                "lm_loss", lm_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "lm_nll_loss", lm_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "lm_ppl", lambda meters: utils.get_perplexity(meters["lm_nll_loss"].avg)
            )
            # log cbmi information
            metrics.log_scalar(
                "mean_cbmi", mean_cbmi_sum / 8 , ntokens, round=3
            )
            metrics.log_scalar(
                "std_cbmi", std_cbmi_sum / 8, ntokens, round=3
            )
            metrics.log_scalar(
                "max_weight", max_weight_sum / 8, ntokens, round=3
            )
            metrics.log_scalar(
                "min_weight", min_weight_sum / 8, ntokens, round=3
            )
            metrics.log_scalar(
                "zero_rate", zero_rate_sum / 8, ntokens, round=3
            )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
            
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True