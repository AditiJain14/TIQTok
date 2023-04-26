import torch
import copy
from typing import List
from tqdm import tqdm

from fairseq import hub_utils
from fairseq.models.transformer import TransformerModel

ckpt = '/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.2_0.2_0.3/checkpoints'
# ckpt = '/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/single_path_latency_0.3/checkpoints'

ckpt_name = "average-model.pt"
data = "/cs/natlang-expts/aditi/mma_runs/data/vi_en/data-bin"

lm = TransformerModel.from_pretrained(ckpt, ckpt_name, data)

tfile = open("/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.2_0.2_0.3/results/pred.translation", "r")
data = tfile.read()
sentences = data.split("\n")
tfile.close()

with open("/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.2_0.2_0.3/analysis/lm_scores.txt", "w") as df:
    for sen in tqdm(sentences):
        the_whole_thing, ll_probs, hypos = lm.sample_with_score([sen], beam=1, verbose=True, sampling=True)
        df.write("S-\t{}\n".format(sen))
        df.write("H-\t{}\n".format(hypos))
        df.write("L-\t{}\n".format(ll_probs))