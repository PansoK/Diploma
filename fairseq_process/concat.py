import argparse
import os
import sys

MAX = sys.maxint

def Concatenate(out_file, bookcorpus, wiki, mil_lines):
    if not os.path.isdir(os.path.dirname(out_file)):
        print('Making folder {}'.format(os.path.dirname(out_file)))
        os.mkdir(os.path.dirname(out_file))
    if os.path.exists(out_file):
        print('Deleting {}'.format(out_file))
        os.remove(out_file)

    print('Concatenating in {}...'.format(out_file))
    # change mil_lines to the correct integer
    if mil_lines.isdigit():
        mil_lines = int(mil_lines)
    else:
        mil_lines = MAX
    with(open(out_file, 'a')) as out:
        print('Appending {}'.format(bookcorpus))
        with(open(bookcorpus, 'r')) as inF:
            for i, s in enumerate(inF):
                out.write(s)
                if i > (mil_lines-1)*10**6:
                    break
        print('Appending {}'.format(wiki))
        with(open(wiki, 'r')) as inF:
            for s in inF:
                out.write(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--mil_lines', metavar='mil_lines', required=True, type=str, default='full', help='mil lines of corpus, else full')
    args = parser.parse_args()
    Concatenate('/data/scratch/panso014/language_modeling/book_plus_wiki/{}_mil_lines/lm_coprus_{}_mil.raw'.format(args.mil_lines, args.mil_lines), \
        '/data/scratch/panso014/language_modeling/bookcorpus/distinct_sentences.txt', \
        '/data/scratch/panso014/language_modeling/wikitext-103/wikitext-103-raw/wiki.train.raw',
        args.mil_lines)
