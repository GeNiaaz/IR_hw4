#!/usr/bin/python3

"""
A0181059A
Abhijit Ravichandran
e0302456@u.nus.edu

CS3245 Homework 2
- Indexing
"""

import re
import nltk
import sys
import getopt
# ------------
import os
import glob
import string
import pickle
import math

MEMORY_LIMIT = 250000

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

"""
Short function to read every file in a folder in sorted order.
"""
def filename_sort(filename):
    number_regex = re.compile(r'(\d+)')
    parts = number_regex.split(filename)
    parts[1::2] = map(int, parts[1::2])
    return parts

"""
Pickles each block, and dumps it in a unique text file.
"""
def write_block(block, block_count):
    # sort each block alphabetically, by token
    block = dict(sorted(block.items(), key=lambda item: item[0]))
    # save each block in 'blocks/' folder
    block_filename = "blocks/block" + str(block_count) + "postings.txt"

    block_length = 0
    with open(block_filename, 'wb') as pickle_block:
        # key = token
        # value = list of docIds
        # pickle each [key, docId] pair and dump
        for key, value in block.items():
            pickle.dump([key, value], pickle_block)
            block_length += 1

    return block_length

"""
Store all text data as blocks, as per the memory limit.
"""
def invert_blocks(in_dir):
    # maintain list of all documents
    all_docs = []
    # maintain size (in bytes) of all blocks
    block_lengths = []

    block = {}
    block_count = 1
    Porter = nltk.stem.PorterStemmer()

    # initialize "blocks/" folder, remove if exists
    if os.path.isdir("blocks"):
        files = glob.glob('blocks/*')
        for f in files:
            os.remove(f)
    else:
        os.makedirs("blocks")

    # read each file, in numerical order
    for filename in sorted(os.listdir(in_dir), key=filename_sort):
        with open(in_dir + "/" + os.fsdecode(filename), encoding="utf-8", errors="ignore", mode="r") as file:
            # avoid Mac OS errors
            if filename != ".DS_Store":
                all_docs.append(int(filename))
                for line in file:
                    for word in nltk.word_tokenize(line):
                        # if memory limit has been reached:
                        if sys.getsizeof(block) >= MEMORY_LIMIT:
                            # write block to temporary block file
                            block_length = write_block(block, block_count)
                            # initialize new block, discarding data that has been written to disk
                            block = {}
                            block_count += 1
                            # store size of block in bytes, to be used in merging
                            block_lengths.append(block_length)

                        # if word is not a number or word isn't all punctuation:
                        if not all(letter.isdigit() or letter in string.punctuation for letter in word):
                            # stem word, add word to block if doesn't yet exist
                            word = Porter.stem(word.lower())
                            if word not in block.keys():
                                block[word] = []

                            # add current document ID to word's docIds list in block
                            value = int(filename)
                            if value not in block[word]:
                                block[word].append(value)

    # write last block to disk
    block_length = write_block(block, block_count)
    # store block size
    block_lengths.append(block_length)

    return [block_lengths, all_docs]

"""
When merged data from blocks exceeds memory limit, generate skip pointers for data,
dump data to disk and store pointers to data in dictionary on memory.
"""
def write_postings(out_postings, merged_postings, dictionary):
    # skip pointer generation
    for key, value in merged_postings.items():
        skips = []
        postings_size = len(value)
        # if len(docIds) <= 2, don't bother with skip pointers
        if postings_size >= 3:
            # create a skip pointer for every sqrt(n) values
            for i in range(0, postings_size+1, math.floor(math.sqrt(postings_size))):
                if (postings_size - i) < math.ceil(math.sqrt(postings_size)):
                    skips.append(len(value) - 1)
                    break
                else:
                    skips.append(i)

        # pickle, and write postings to disk
        with open(out_postings, 'ab') as postings_file:
            # save pointer to data in dictionary
            dictionary[key] = [postings_size, postings_file.tell()]
            postings_file.write(pickle.dumps([key, value, skips]))
            dictionary[key].append(postings_file.tell() - dictionary[key][1])
            # empty postings data on memory
            merged_postings = {}

"""
Carries out a n-way merge of data from all blocks on disk.
"""
def merge_blocks(block_lengths, all_docs, out_dict, out_postings):
    block_count = len(block_lengths)

    # initialize dictionary and postings list
    # postings lists to be written to disk when exceeds memory limit
    dictionary = {}
    merged_postings = {}

    # initialize list of block filenames, and unpickle data
    block_filenames = ["blocks/block" + str(i) + "postings.txt" for i in range(1, block_count+1)]
    block_files = [pickle.Unpickler(open(i, "rb")) for i in block_filenames]

    # list of a line of data from each block
    # each line is one token, and its docIds
    block_reads = [block.load() for block in block_files]
    block_loads = [0 for block in block_files]

    # if dictionary and postings files already exist: delete and re-index
    if os.path.exists(out_dict):
        os.remove(out_dict)
    if os.path.exists(out_postings):
        os.remove(out_postings)

    # initiate n-way merge
    while block_files:
        # if postings list size exceeds memory limit
        if sys.getsizeof(merged_postings) >= MEMORY_LIMIT:
            # write to disk
            write_postings(out_postings, merged_postings, dictionary)
            # empty postings file on memory
            merged_postings = {}

        # else: keep reading data from blocks
        try:
            # get token that is alphabetically first, among each token read from each block
            earliest_word = min([item[0] for item in block_reads])
            if earliest_word not in merged_postings.keys():
                # store token and its complement in postings file in memory
                merged_postings[earliest_word] = []
                merged_postings["~" + earliest_word] = all_docs
            for i in range(0, len(block_reads)):
                key = block_reads[i][0]
                val = block_reads[i][1]
                # only if the block has docId data on the alphabetically 'first' token, retrieve that data
                # then, load next data line from that block
                # else, skip it
                if key == earliest_word:
                    merged_postings[key].extend(val)
                    merged_postings[key].sort()
                    # maintain complement to word in postings
                    merged_postings["~" + key] = ([doc_id for doc_id in merged_postings["~" + key] if (doc_id not in val)])
                    merged_postings["~" + key].sort()
                    # load next line of data from block
                    block_reads[i] = block_files[i].load()
                    block_loads[i] += 1

        # if entire block has been read:
        except EOFError:
            pop_indexes = []
            for i in range(0, len(block_lengths)):
                if block_loads[i] == block_lengths[i]-1:
                    pop_indexes.append(i)

            # remove that block from list of blocks to be read
            for i in reversed(pop_indexes):
                block_files.pop(i)
                block_lengths.pop(i)
                block_loads.pop(i)
                block_reads.pop(i)

    # write last postings file to disk
    write_postings(out_postings, merged_postings, dictionary)
    merged_postings = {}

    # pickle entire dictionary and save to disk
    with open(out_dict, 'wb') as pickle_dict:
        pickle.dump(dictionary, pickle_dict)

"""
Main function to call invert_blocks, and afterwards, merge_blocks.
Build index from documents stored in the input directory,
then output the dictionary file and postings file
"""
def build_index(in_dir, out_dict, out_postings):
    print('indexing...')
    block_lengths, all_docs = invert_blocks(in_dir)
    print('merging...')
    merge_blocks(block_lengths, all_docs, out_dict, out_postings)

    print("\ndone.")
    print(out_dict, "and", out_postings, "generated.")
    print("temporary merge files are stored in 'blocks/' folder.")


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
