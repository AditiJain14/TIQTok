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
    ROOT="path/to/working/dir"

    DATA="${ROOT}/data/vi_en/data-bin"

    EXPT="${ROOT}/experiments/en_vi"
    mkdir -p ${EXPT}

    FAIRSEQ="${ROOT}/mma"

    USR="./examples/simultaneous_translation"

    lambda=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_latency_0.1_0.2_${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 50 \
    --arch transformer_monotonic_iwslt_de_en \
    --user-dir "${USR}" \
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
    --criterion latency_augmented_label_smoothed_cross_entropy_cbmi \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 4500 --update-freq 3 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 25 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 3000 \
    --token-scale 0.1 --sentence-scale 0.2 \
    --wandb-project LM_Adaptive_EnVi \
    --empty-cache-freq 45 --max-epoch 50\
    | tee -a ${TBOARD}/train_log.txt

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

The numerical results on IWSLT15 English-to-Vietnamese:

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

If this repository is useful for you, please cite as:

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
