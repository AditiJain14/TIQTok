
ROOT="/cs/natlang-expts/aditi/mma_runs"
# ROOT="path/to/working/dir"

DATA="${ROOT}/data/vi_en/data-bin"

NKROOT="/local-scratch/nishant/simul/mma_runs/"

EXPT="${NKROOT}/experiments/en_vi"
mkdir -p ${EXPT}

FAIRSEQ="${NKROOT}/mma"

USR="./examples/simultaneous_translation"




export CUDA_VISIBLE_DEVICES=0,1


gma_lm(){
    delta=$1
    # name="single_path_latency_${lambda}"
    name="lmloss_gma_${delta}"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/nishant_gma/${name}/checkpoints"
    TBOARD="${EXPT}/nishant_gma/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 50 \
    --arch transformer_iwslt_de_en_gma \
    --uni-encoder True \
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
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --delta ${delta} \
    --save-dir $CKPT \
    --max-tokens 2667 --update-freq 3 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 20 \
    --add-language-model --pretrain-steps 4000\
    --share-lm-decoder-softmax-embed \
    --token-scale 0.1 --sentence-scale 0.1 \
    --empty-cache-freq 45 --max-epoch 38\
    --eval-bleu \
    --eval-bleu-args '{"beam": 1}' \
    --wandb-project LM_Adaptive_EnVi \
    | tee -a ${TBOARD}/train_log.txt
    # --tensorboard-logdir ${TBOARD} \
# 
    #dont use cbmi loss for getting checkpoints for lambda>0.1, set pretrain-steps high. 
    #This will also train LM decoder with rate lm_rate*10
    #load this checkpoint for lambda>0.1 runs 
    #--restore-file ""\

}

gma(){
    delta=$1
    # name="single_path_latency_${lambda}"
    name="gma_${delta}_temp"
    export WANDB_NAME="${name}"

    CKPT="${EXPT}/nishant_gma/${name}/checkpoints"
    TBOARD="${EXPT}/nishant_gma/${name}/logs"
    mkdir -p ${CKPT} ${TBOARD}

    python ${FAIRSEQ}/train.py --ddp-backend=no_c10d ${DATA} \
    --source-lang en --target-lang vi \
    --log-format simple --log-interval 100 \
    --arch transformer_iwslt_de_en_gma \
    --uni-encoder True \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --weight-decay 0.0001 \
    --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --max-update 100000 \
    --left-pad-source \
    --single-path \
    --dual-weight 0.0 \
    --delta ${delta} \
    --save-dir $CKPT \
    --max-tokens 8000 --update-freq 1 \
    --required-batch-size-multiple 8 \
    --best-checkpoint-metric "ppl" \
    --keep-last-epochs 15 \
    --empty-cache-freq 45 --max-epoch 45 \
    --fp16 \
    | tee -a ${TBOARD}/train_log.txt

    # --tensorboard-logdir ${TBOARD} \
        # --wandb-project LM_Adaptive_EnVi \
        # 
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 1}' \
    #dont use cbmi loss for getting checkpoints for lambda>0.1, set pretrain-steps high. 
    #This will also train LM decoder with rate lm_rate*10
    #load this checkpoint for lambda>0.1 runs 
    #--restore-file ""\

}

###############################################
export CUDA_VISIBLE_DEVICES=0,1

# mma_il 0.3
# train_lm_only 0
# mma_h
# mma_wait_k
# wait_info
# wait_info_adaptive_ft
# mma_il_freezelmchkpt 0.1
# wait_info_adaptive_train
# mma_il_with_pretrained 0.4
# mma_il_lm 0.1
# gma_lm 1

gma 1