#!/usr/bin/env bash
#

# set path
ROOT="/cs/natlang-expts/aditi/mma_runs"
FAIRSEQ="${ROOT}/mma"

TEXT="${ROOT}/data/vi_en"

RAW="${TEXT}/raw"
mkdir -p $RAW

# download data
# for lang in "en" "vi"; do
#     # train, dev, test
#     wget -O train.$lang -P "${RAW}/" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.${lang}" 
#     wget -O valid.$lang -P "${RAW}/" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.${lang}"
#     wget -O test.$lang -P "${RAW}/" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.${lang}"
#     # vocab files
#     wget -P "${RAW}/" "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/vocab.${lang}"
#     # fix dict file for fairseq
#     # strip the first three special tokens and append fake counts for each vocabulary
#     tail -n +4 "${RAW}/vocab.${lang}" | cut -f1 | sed 's/$/ 100/g' > "${RAW}/dict.${lang}"
# done


# Preprocess/binarize the data
mkdir -p "$TEXT/data-bin"

python3 ${FAIRSEQ}/fairseq_cli/preprocess.py --source-lang vi --target-lang en \
    --trainpref $RAW/train \
    --validpref $RAW/valid \
    --testpref $RAW/test \
    --destdir $TEXT/data-bin \
    --tgtdict $TEXT/raw/dict.en \
    --srcdict $TEXT/raw/dict.vi \
    --workers 20

