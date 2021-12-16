#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
from collections import Counter
from itertools import zip_longest
from multiprocessing import Pool

from fairseq import options_split, tasks, utils
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset
from fairseq.file_chunker_utils import find_offsets

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")


def main(args):
    utils.import_user_module(args)

    ###############################
    if args.train_split:
        assert (args.align_suffix == None and args.alignfile == None and args.dataset_impl != "raw" and 0.0 < args.train_split < 1.0)
        os.makedirs(args.destdir+"1", exist_ok=True)  # make destination folder
        os.makedirs(args.destdir+"2", exist_ok=True)  # make destination folder

    else:
        os.makedirs(args.destdir, exist_ok=True)  # make destination folder
    ###############################
    # logging info goes to the first destdir
    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir + ("1" if args.train_split else ""), "preprocess.log"),
        )
    )

    logger.info(args)

    assert args.dataset_impl != "huffman", "preprocessing.py doesn't support Huffman yet, use HuffmanCodeBuilder directly."

    task = tasks.get_task(args.task)  # translation in pic

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")  # set to 'examples/translation/iwslt14.tokenized.de-en/train'

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    ###############################
    def dest_path(prefix, lang):
        if args.train_split:
            return [os.path.join(args.destdir+ "1", file_name(prefix, lang)),
                        os.path.join(args.destdir+ "2", file_name(prefix, lang))]
        else:
            return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):  # save source and target lang dicts in fairseq-data-bin/dataset.tokenized.{src}-{trg}
        if args.train_split:
            dst_paths = dest_path("dict", lang)
            return [dst_paths[0] + ".txt", dst_paths[1] + ".txt"]
        else:
            return dest_path("dict", lang) + ".txt"
    ###############################

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    target = not args.only_source  # False in pic

    ###############################
    if args.train_split:
        if not args.srcdict:
            paths = dict_path(args.source_lang)
            if os.path.exists(paths[0]):  # srcdict False and os.exists False
                raise FileExistsError(paths[0])
            if os.path.exists(paths[1]):  # srcdict False and os.exists False
                raise FileExistsError(paths[1])
        if target and not args.tgtdict:
            paths = dict_path(args.target_lang)
            if os.path.exists(paths[0]):
                raise FileExistsError(paths[0])
            if os.path.exists(paths[1]):
                raise FileExistsError(paths[1])
    else:
        if not args.srcdict and os.path.exists(dict_path(args.source_lang)):  # srcdict False and os.exists False
            raise FileExistsError(dict_path(args.source_lang))
        if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
            raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:  # False in pic
        assert (
            not args.srcdict or not args.tgtdict
        ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]},
                src=True,
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:  # srcdict False in pic
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert (
                args.trainpref  # set to 'examples/translation/iwslt14.tokenized.de-en/train'
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)  # building task dictionary for the language

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    # HERE is one: these should be saved in every folder I have splitted the dataset
    ###############################
    if args.train_split:
        dict_paths = dict_path(args.source_lang)
        src_dict.save(dict_paths[0])
        src_dict.save(dict_paths[1])
    else:
        src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        if args.train_split:
            dict_paths = dict_path(args.target_lang)
            tgt_dict.save(dict_paths[0])
            tgt_dict.save(dict_paths[1])
        else:
            tgt_dict.save(dict_path(args.target_lang))
    ###############################

    if args.dict_only:
        return

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, folder=None):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = find_offsets(input_file, num_workers)  # given a file and a number of chuncks, find the offsets in the file 
        # to be able to chunk around full lines
        # file is splitted into num_workers offsets
        (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id, (start_offset, end_offset) in enumerate(
                more_chunks, start=1
            ):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        start_offset,
                        end_offset,
                        folder,
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin", folder),
            impl=args.dataset_impl,
            vocab_size=len(vocab),
        )
        merge_result(
            Binarizer.binarize(
                input_file,
                vocab,
                lambda t: ds.add_item(t),
                offset=first_chunk[0],
                end=first_chunk[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang, folder)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx", folder))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result["nseq"]

        input_file = input_prefix
        offsets = find_offsets(input_file, num_workers)
        (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id, (start_offset, end_offset) in enumerate(
                more_chunks, start=1
            ):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        start_offset,
                        end_offset,
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, None, "bin"), impl=args.dataset_impl
        )

        merge_result(
            Binarizer.binarize_alignments(
                input_file,
                utils.parse_alignment,
                lambda t: ds.add_item(t),
                offset=first_chunk[0],
                end=first_chunk[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info("[alignments] {}: parsed {} alignments".format(input_file, nseq[0]))

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, folder=None):
        if args.dataset_impl == "raw":  # was set to mmap in pic
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, folder)

    def make_all(lang, vocab, indices_big=None, indices_small=None):
        if args.trainpref:
            ###############################
            if args.train_split:
                print(indices_big[0:2])
                print(indices_small[0:2])
                assert (indices_big != None and indices_small != None)
                # create files
                with open(train_path(lang), 'r') as f_train:
                    lines = f_train.readlines()
                lines_big = [lines[i] for i in indices_big]
                lines_small = [lines[i] for i in indices_small]
                path_big = "{}{}".format(args.trainpref, ("1." + lang))
                if os.path.exists(path_big):
                    logger.info("found big training file {}, deleting it".format(path_big[-9:]))
                    os.remove(path_big)
                with open(path_big, 'a') as f_big:
                    for line in lines_big:
                        f_big.write(line)
                path_small = "{}{}".format(args.trainpref, ("2." + lang))
                if os.path.exists(path_small):
                    logger.info("found small training file {}, deleting it".format(path_small[-9:]))
                    os.remove(path_small)
                with open(path_small, 'a') as f_small:
                    for line in lines_small:
                        f_small.write(line)
                # make the two datasets
                make_dataset(vocab, args.trainpref + "1", "train", lang, num_workers=args.workers, folder="1")
                make_dataset(vocab, args.trainpref + "2", "train", lang, num_workers=args.workers, folder="2")
            else:
                make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)  # vocal of source, path to train file, _, source_lang
            ###############################
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(
                    vocab, validpref, outprefix, lang, num_workers=args.workers, folder="1"
                )
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers, folder="1")


    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.trainpref + "." + args.align_suffix,
                "train.align",
                num_workers=args.workers,
            )
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.validpref + "." + args.align_suffix,
                "valid.align",
                num_workers=args.workers,
            )
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.testpref + "." + args.align_suffix,
                "test.align",
                num_workers=args.workers,
            )


    ###############################
    import random
    if args.train_split:
        # create indices for the two files
        num_lines = 0
        with open(train_path(args.source_lang), 'r') as f_train:  # both files have the same number of lines
            num_lines = len(f_train.readlines())
        split = args.train_split if args.train_split > 0.5 else 1-args.train_split  # indices pointing to lines that will belong in the big subset
        indices = list(range(num_lines))
        random.shuffle(indices)
        indices_big = indices[:int(num_lines*split)]
        indices_small = indices[int(num_lines*split):]
        make_all(args.source_lang, src_dict, indices_big, indices_small)
        if target:
            make_all(args.target_lang, tgt_dict, indices_big, indices_small)
        # only wrote them to the first one, now copying them to the second
        if args.validpref:
            copy_to_dest2(args, "valid")
        if args.testpref:
            copy_to_dest2(args, "test")
    else:
        make_all(args.source_lang, src_dict)
        if target:
            make_all(args.target_lang, tgt_dict)
        if args.align_suffix:
            make_all_alignments()

    if args.train_split:
        logger.info("Wrote preprocessed data to {} and {}".format(args.destdir+"1", args.destdir+"2"))
    else:
        logger.info("Wrote preprocessed data to {}".format(args.destdir))
    ###############################

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding="utf-8") as align_file:
            with open(src_file_name, "r", encoding="utf-8") as src_file:
                with open(tgt_file_name, "r", encoding="utf-8") as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
            os.path.join(
                args.destdir,
                "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
            ),
            "w",
            encoding="utf-8",
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, folder=None, append_eos=True):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin", folder),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx", folder))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(
        filename, parse_alignment, consumer, offset=offset, end=end
    )
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang, folder=None):
    ###############################
    #print(folder)
    if args.train_split:
        assert (folder is not None)
        base = "{}/{}".format(args.destdir + folder, output_prefix)
    else:
        base = "{}/{}".format(args.destdir, output_prefix)
    ###############################
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension, folder=None):
    base = dataset_dest_prefix(args, output_prefix, lang, folder)
    return "{}.{}".format(base, extension)


def copy_to_dest2(args, prefix):
    files = os.listdir(args.destdir + "1")
    files_to_copy = [file for file in files if file[:2] == prefix[:2]]
    logger.info("Copying {} to folder 2".format(files_to_copy if len(files_to_copy) > 1 else files_to_copy[0]))
    for file in files_to_copy:
        shutil.copy(args.destdir + "1/" + file, args.destdir + "2")


def cli_main():
    parser = options_split.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
