
ROOT="/cs/natlang-expts/aditi/mma_runs"
# ROOT="path/to/working/dir"

DATA="${ROOT}/data/vi_en/data-bin"

EXPT="${ROOT}/experiments/vi_en"
mkdir -p ${EXPT}

FAIRSEQ="${ROOT}/mma"

USR="./examples/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0,1

# infinite lookback
mma_il_legacy(){
    lambda=$1
    name="data_redo_latency_${lambda}"

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
    --criterion latency_augmented_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 180000 \
    --latency-weight-avg  ${lambda} \
    --noise-var 1.5 \
    --left-pad-source \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 6000 --update-freq 2 --reset-optimizer \
    --tensorboard-logdir ${TBOARD} \
    | tee -a ${TBOARD}/train_log.txt

}

mma_il(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="single_path_lm_latency_${lambda}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} --no-save \
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
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3000 --update-freq 6 \
    --best-checkpoint-metric "ppl" \
    --patience 10 --keep-last-epochs 12 \
    --tensorboard-logdir ${TBOARD} --wandb-project MMAH \
    --add-language-model \
    --share-lm-decoder-softmax-embed
    #| tee -a ${TBOARD}/train_log.txt

}
#mma-il with lm 
mma_il_lm(){
    lambda=$1
    # name="single_path_latency_${lambda}"
    name="single_path_lmloss_latency_${lambda}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/infinite/${name}/checkpoints"
    TBOARD="${EXPT}/infinite/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py  --ddp-backend=no_c10d ${DATA} --no-save \
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
    --single-path \
    --dual-weight 0.0 \
    --save-dir $CKPT \
    --max-tokens 3000 --update-freq 6 \
    --best-checkpoint-metric "ppl" \
    --patience 10 --keep-last-epochs 12 \
    --add-language-model \
    --share-lm-decoder-softmax-embed \
    --pretrain-steps 1500 \
    --token-scale 0.1 --sentence-scale 0.1 \
    #| tee -a ${TBOARD}/train_log.txt
    
    --wandb-project MMAH
    # --tensorboard-logdir ${TBOARD} \

}

# hard align
mma_h(){
    CKPT="${EXPT}/checkpoints/hard"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type hard_aligned \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d
}


# monotnic wait k
mma_wait_k(){
    CKPT="${EXPT}/checkpoints/waitk"
    mkdir -p ${CKPT}

    fairseq-train \
    $DATA \
    --log-format simple --log-interval 100 \
    --source-lang de --target-lang en \
    --task translation \
    --simul-type waitk \
    --waitk-lagging 3 \
    --user-dir $USR \
    --mass-preservation \
    --criterion latency_augmented_label_smoothed_cross_entropy_pure \
    --latency-weight-avg  0.1 \
    --max-update 50000 \
    --arch transformer_monotonic_iwslt_de_en --save-dir $CKPT \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7  --warmup-updates 4000 \
    --lr 5e-4 --min-lr 1e-9 --clip-norm 0.0 --weight-decay 0.0001\
    --dropout 0.3 \
    --label-smoothing 0.1\
    --max-tokens 7000 \
    --ddp-backend no_c10d
}

wait_info(){
    name="waitinfo_maxk8"
    CKPT="${EXPT}/${name}/checkpoints"
    TBOARD="${EXPT}/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    echo "Training Wait-Info.."

    python3 ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} --arch transformer_iwslt_de_en \
    --source-lang de --target-lang en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --log-format simple \
    --log-interval 100 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --left-pad-source False \
    --save-dir ${CKPT} \
    --max-tokens 8192 --update-freq 1 \
    --max-update 60000 \
    --fp16 \
    --keep-last-epochs 10 \
    --best-checkpoint-metric "ppl" \
    --patience 10 \
    --tensorboard-logdir ${TBOARD} | tee -a ${TBOARD}/train_log.txt

}

wait_info_adaptive_train(){
    name="expt_adapt_train_T1.75"
    CKPT="${EXPT}/${name}/checkpoints"
    TBOARD="${EXPT}/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    TGT_DICT="${DATA}/dict.en.txt"

    echo "Training Wait-Info + Adaptive .."

    python3 ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} --arch transformer_iwslt_de_en \
    --source-lang de --target-lang en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --log-format simple \
    --log-interval 100 \
    --dropout 0.3 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy_exponential_adapt \
    --lr-scheduler inverse_sqrt \
    --lr 5e-4 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --adaptive-training \
    --dict-file ${TGT_DICT} \
    --adaptive-method 'exp' \
    --adaptive-T 1.75 \
    --weight-drop 0.3 \
    --weight-decay 0.0 \
    --label-smoothing 0.1 \
    --left-pad-source False \
    --save-dir ${CKPT} \
    --max-tokens 8192 --update-freq 1 \
    --max-update 60000 \
    --fp16 \
    --keep-last-epochs 10 \
    --best-checkpoint-metric "ppl" \
    --patience 10 \
    --tensorboard-logdir ${TBOARD} 
    #| tee -a ${TBOARD}/train_log.txt

}

wait_info_adaptive_ft(){
    RESTORE="${EXPT}/base/checkpoints/checkpoint30.pt"

    CKPT="${EXPT}/expt_adapt_0.1/checkpoints"
    TBOARD="${EXPT}/expt_adapt_0.1/logs"
    mkdir -p ${CKPT} ${TBOARD}

    TGT_DICT="${DATA}/dict.en.txt"

    echo "Fine-tuning Wait-Info + Adaptive .."

    python3 ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} --arch transformer_iwslt_de_en \
    --source-lang de --target-lang en \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --log-format simple \
    --log-interval 100 \
    --dropout 0.3 \
    --encoder-attention-heads 8 \
    --decoder-attention-heads 8 \
    --criterion label_smoothed_cross_entropy_exponential_adapt \
    --lr-scheduler inverse_sqrt \
    --lr 0.00028 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 1000 \
    --reset-optimizer --reset-lr-scheduler \
    --adaptive-training \
    --dict-file ${TGT_DICT} \
    --adaptive-method 'exp' \
    --adaptive-T 1.75 \
    --weight-drop 0.1 \
    --weight-decay 0.0 \
    --label-smoothing 0.1 \
    --left-pad-source False \
    --restore-file ${RESTORE} \
    --save-dir ${CKPT} \
    --max-tokens 8192 --update-freq 1 \
    --max-update 15000 \
    --fp16 \
    --save-interval-updates 1000 \
    --keep-interval-updates 2 \
    --tensorboard-logdir ${TBOARD} 
    #| tee -a ${TBOARD}/train_log.txt

}



###############################################
export CUDA_VISIBLE_DEVICES=0,1

# mma_il 0.1
# mma_h
# mma_wait_k
# wait_info
# wait_info_adaptive_ft

# wait_info_adaptive_train

mma_il_lm 0.1