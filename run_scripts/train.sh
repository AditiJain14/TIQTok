
ROOT="/cs/natlang-expts/aditi/TIQTok"
# ROOT="path/to/working/dir"

# DATA="${ROOT}/data/vi_en/data-bin"
DATA="/cs/natlang-expts/aditi/mma_runs/data/vi_en/data-bin"
EXPT="${ROOT}/experiments/en_vi"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/mma"

USR="./examples/simultaneous_translation"

export CUDA_VISIBLE_DEVICES=0

mma_il_lm(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_latency_0.2_0.2_${lambda}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} \
    --source-lang vi --target-lang en \
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
    --save-dir $CKPT \
    --max-tokens 10000 --update-freq 2 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 15 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 3000 \
    --token-scale 0.2 --sentence-scale 0.2 \
    --empty-cache-freq 45 --max-epoch 45\
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \
        # --restore-file "/cs/natlang-expts/aditi/mma_runs/experiments/vi_en/infinite/lmloss_latency_0.1_0.1forchkpt_0/checkpoints/checkpoint8.pt" \

    #dont use cbmi loss for getting checkpoints for lambda>0.1, set pretrain-steps high. 
    #This will also train LM decoder with rate lm_rate*10
}

mma_il_lm 0.3