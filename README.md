# Language Model Based Target Token Importance Rescaling for Simultaneous Neural Machine Translation
Code implementation of our IWSLT 2023 [paper](https://aclanthology.org/2023.iwslt-1.32/), implemented with the open-source toolkit [Fairseq](https://github.com/pytorch/fairseq).



## Requirements and Installation

- Python version = 3.6

- [PyTorch](http://pytorch.org/) version = 1.7

- Install fairseq:

  ```bash
  git clone https://github.com/AditiJain14/TIQTok.git
  cd TIQTok/mma
  pip install --editable ./
  ```



## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding `--joined-dictionary` for WMT15 German-English:

```bash
src=SOURCE_LANGUAGE
tgt=TARGET_LANGUAGE
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA
data=PATH_TO_DATA

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training

Train the TIQ SiMT with the following command:

- For IWSLT15 English-Vietnamese: we set ***latency weight*** = 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55.
- For other datasets, we use ***latency weight*** = 0.01, 0.1, 0.2, 0.3, 0.4.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
lambda=LATENCY_WEIGHT

python train.py  --ddp-backend=no_c10d ${data} --arch transformer_monotonic_iwslt_de_en --share-all-embeddings \
    --user-dir ./examples/simultaneous_translation \
    --simul-type infinite_lookback \
    --mass-preservation \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --max-update 180000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --dual-weight 1.0 \
    --save-dir ${modelfile} \
    --max-tokens 2400 --update-freq 4
```

### Inference

Evaluate the model with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
ref=PATH_TO_REFERENCE

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python fairseq_cli/generate.py ${data} \
    --path ${modelfile}/average-model.pt  \
    --left-pad-source \
    --batch-size 250 \
    --beam 1 \
    --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref} < pred.translation
```



## Our Results

The numerical results on WMT15 German-to-English:

| **latency weight** | **AP** | **AL** | **DAL** | **BLEU** |
| :----------------: | :----: | :----: | :-----: | :------: |
|        0.55        |  0.66  |  3.10  |  5.12   |  28.60   |
|        0.5         |  0.67  |  3.60  |  5.78   |  28.81   |
|        0.3         |  0.68  |  3.86  |  6.12   |  28.90   |
|        0.2         |  0.71  |  4.58  |  7.22   |  28.74   |
|        0.1         |  0.74  |  5.34  |  8.18   |  28.65   |
|        0.01        |  0.89  |  9.89  |  14.37  |  28.67   |

More results please refer to the paper.



## Citation

In this repository is useful for you, please cite as:

```
@inproceedings{jain-etal-2023-language,
    title = "Language Model Based Target Token Importance Rescaling for Simultaneous Neural Machine Translation",
    author = "Jain, Aditi  and
      Kambhatla, Nishant  and
      Sarkar, Anoop",
    editor = "Salesky, Elizabeth  and
      Federico, Marcello  and
      Carpuat, Marine",
    booktitle = "Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)",
}

```
