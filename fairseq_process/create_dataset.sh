#! /bin/sh

# A script to run all models

mil_lines=15
mkdir ${mil_lines}_mil_lines
python concat.py --mil_lines $mil_lines

mkdir -p ${mil_lines}_mil_lines/gpt2_bpe
wget -O ${mil_lines}_mil_lines/gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O ${mil_lines}_mil_lines/gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

cd /home/panso014/diploma/fairseq/
python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/gpt2_bpe/encoder.json \
        --vocab-bpe /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/gpt2_bpe/vocab.bpe \
        --inputs /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/lm_coprus_${mil_lines}_mil.raw \
        --outputs /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/lm_coprus_${mil_lines}_mil.bpe \
        --keep-empty \
        --workers 60

cd /data/scratch/panso014/language_modeling/book_plus_wiki/
wget -O ${mil_lines}_mil_lines/gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
fairseq-preprocess \
    --only-source \
    --srcdict /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/gpt2_bpe/dict.txt \
    --trainpref /data/scratch/panso014/language_modeling/book_plus_wiki/${mil_lines}_mil_lines/lm_coprus_${mil_lines}_mil.bpe \
    --validpref /data/scratch/panso014/language_modeling/wikitext-103/wikitext-103-raw/wiki.valid.bpe \
    --testpref /data/scratch/panso014/language_modeling/wikitext-103/wikitext-103-raw/wiki.test.bpe \
    --destdir /data/scratch/panso014/fairseq-data-bin/book_plus_wiki/${mil_lines}_mil_lines \
    --workers 60
