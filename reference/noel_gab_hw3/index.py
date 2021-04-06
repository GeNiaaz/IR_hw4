#!/usr/bin/python3
# IR STD IMPORTS
import re
import nltk
import sys
import getopt

# Filesystem imports
from os import listdir
from os.path import join

# Data (de)serialization
import json
import linecache
import pickle

# Misc
import math
import copy

# Store
import string
import shutil

#### dictionary.txt FORMAT
# {<word>: {idf_weight: idf, curr_id: doc_id, start: x, length: y}, ...}
# idf = log(N/df) [no normalization]
# curr_id tracks the doc_id to decide if df should be updated
# start, length gives us posting ref to postings file

#### BLOCK IN MEM FORMAT (current_block)
# {<word>: postings, ...}
# When we hit an arbitrary limit for doc_ids, we need to write the block to disk

#### POSTINGS FORMAT
# postings := { <doc_id>: {tf_weight_norm: tf, next_doc_id: a, skip_doc_id: b}, ... } # sorted by doc_id
# tf = 1 + log(term_freq) [followed by normalization]

#### BLOCK ON DISK FORMAT
# <postings><postings><postings>...
# Access by seek(start), read(length) (supplied by dictionary.txt mapping)

#### Toggling blocks
# EVENT: Max block (1)
# block_files = [prev_file, cur_file]
# write(cur_block_1, prev_file)
# EVENT: Max block (2)
# merge(prev_file, cur_block_2, cur_file)
# block_files[0] = cur_file
# block_files[1] = prev_file

#### WRITE a word's postings
#### MUT VAR: START (Used to track where we are in the block_file)
####                (Always updated after writing)
# postings = block[word]
# postings_ser = pickle.dumps(postings) # convert into binary format
# file.seek(start)
# length = file.write(postings_ser) # write returns length
# dictionary[word]['start'] = start; dicitonary[word]['length'] = length
# start = start + length

#### ACCESSING A WORD's postings
# start = dictionary.get(word)['start']
# length = dictionary.get(word)['length']
#
# file.seek(start) -> moves our file pointer to start of where the postings for word begins
#
# postings_ser = file.read(length) -> reads the entire binary serialized postings (for the word) into mem.
# ^ Binary format
#
# postings = pickle.loads(postings_ser)

### SUBBLOCKS (Implemented as hashmap Linked List (LL) for skipping)
# postings := { <doc_id>: {tf_weight_norm, next_doc_id, skip_doc_id}, ... } # sorted by doc_id
#
# Example:
# Scenario: we have a word, which exists in documents 100, 1000, 1100, 2000, 2400, 3000
# Interval = 2
# { 100: {tf_weight_norm: 0.01, next_doc_id: 1000, skip_doc_id: 1100}, 1000: {tf_weight_norm: 0.015, next_doc_id: 1100, skip_doc_id: None}, ...}


def build_index(in_dir, out_dict, out_postings):
    block_size = 500 # Arbitrary max number of docs to store into disk
    doc_total_count = 0 # Track number of docs in entire collection
    doc_count = 0 # Track number of docs per block for storing to disk
    current_block = {}

    dictionary = {} 
    existing_block_file = [None, None] # Control var for INIT BLOCK / MERGE BLOCK
    tf_weights = {}

    # Shared stemmer object, do not instantiate per word
    stemmer = nltk.stem.porter.PorterStemmer()

    print("Indexing...")

    for doc_id_str in listdir(in_dir):
        doc_total_count += 1
        doc_id = int(doc_id_str)

        if doc_count > block_size:
            # Store the current block on disk
            tf_weights_copy = copy.deepcopy(tf_weights)
            write_to_block(current_block, existing_block_file, dictionary, update_tf_weights(tf_weights_copy))
            tf_weights_copy.clear()

            # Flush the in-mem block
            current_block.clear()
            doc_count = 0

        doc_count += 1
        doc_path = join(in_dir, doc_id_str)
        with open(doc_path, 'r', encoding='unicode_escape') as document:
            for lines in document.readlines():
                words = nltk.word_tokenize(lines)
                # Store the words on the block
                for word in words:
                    
                    true_words = [word]
                    
                    if word in string.punctuation: # Do not include sole punctuations
                        continue
                    if '/' in word: # Break up words separated by a slash
                        true_words = word.split('/')
                    if '-' in word:  # Break up words separated by a hyphen
                        true_words = word.split('-')

                    for true_word in true_words:
                        ### PROCESS WORD
                        base_word = stemmer.stem(true_word.lower())

                        ### INIT WORD DICT, COUNT DOC FREQ
                        if dictionary.get(base_word) == None: # No base_word <-> positional mapping
                            dictionary[base_word] = {'idf_weight': 1, 'curr_doc': doc_id, 'start': 0, 'length': 0} # idf and positions will be updated later
                        else:
                            if dictionary[base_word]['curr_doc'] != doc_id:
                                dictionary[base_word]['idf_weight'] += 1
                                dictionary[base_word]['curr_doc'] = doc_id

                        ### INIT WORD <-> DOCS
                        if current_block.get(base_word) == None: # Word not stored yet
                            current_block[base_word] = {} # Initialize empty doc_id LL

                        ### INIT DOCS: DOC_ID, TF_WEIGHT
                        if current_block[base_word].get(doc_id) == None: # doc_id LL not initialized
                            current_block[base_word][doc_id] = {'tf_weight_norm': 0, 'next_doc_id': None, 'skip_doc_id': None} 

                        ### COUNT TERM FREQ
                        if tf_weights.get(doc_id) == None:
                            tf_weights[doc_id] = {} 
                        
                        if tf_weights[doc_id].get(base_word) == None:
                            tf_weights[doc_id][base_word] = 1
                        else: 
                            tf_weights[doc_id][base_word] += 1

    # Write whatever is left
    write_to_block(current_block, existing_block_file, dictionary, update_tf_weights(tf_weights))

    # Clear current_block and tf_weights dictionary
    current_block.clear()
    tf_weights.clear()

    # Construct skip postings in dictionary
    construct_skip_postings(dictionary, existing_block_file)

    # Update idf_weights in dictionary
    update_idf_weights(dictionary, doc_total_count)

    # Write dictionary to file
    with open(out_dict, 'wb') as dict_file:
        dict_ser = pickle.dumps(dictionary)
        dict_file.write(dict_ser)

    # Copy latest block (postings)
    shutil.copyfile(existing_block_file[1], out_postings)

#### INTERNALS
def update_tf_weights(tf_weights):
    res = tf_weights
    for doc_id in res:
        sum_of_squares = 0
        
        for word in res[doc_id]:
            res[doc_id][word] = 1 + math.log(res[doc_id][word], 10) # Formula: tf_weight = 1 + log(tf); tf >= 1
            sum_of_squares += (res[doc_id][word])**2
        
        # Find Length[N], the document length 
        doc_length = math.sqrt(sum_of_squares) 

        # Do normalization
        for word in res[doc_id]:
            res[doc_id][word] /= doc_length
    return res

def update_idf_weights(dictionary, collection_size):
    for word in dictionary:
        dictionary[word]['idf_weight'] = math.log(collection_size / dictionary[word]['idf_weight'], 10) # Formula: idf_weight = log(N/df); idf_weight >= 1

def interval_len(length):
    return math.floor(math.sqrt(length))

def construct_skip_postings(dictionary, existing_block_file):
    prev_block_file = existing_block_file[0]
    new_block_file = existing_block_file[1]
    write_start = 0
    # For each word's postings
    for word, position in dictionary.items():
        start = position['start']
        length = position['length']
        postings = None # Rewrite the updated postings which have skip pointers
        with open(prev_block_file, 'rb') as bf:
            # Load postings for a word
            bf.seek(start)
            postings_ser = bf.read(length)
            postings = pickle.loads(postings_ser)

            prev_skip_doc_id = -1 # Skip pointer prev -> current
            prev_doc_id = -1 # Previous doc_id
            interval_count = 0 # Every docs_interval we include a skip pointer
            # We iterate through docs, ordered by doc_id
            docs_interval = interval_len(len(postings))
            for doc_id, doc_meta in postings.items():
                if prev_doc_id == -1: # First doc_id in our postings
                    prev_skip_doc_id = doc_id
                else: # Subsequent doc_id
                    if interval_count == 0:
                        postings[prev_skip_doc_id]['skip_doc_id'] = doc_id
                        prev_skip_doc_id = doc_id

                    postings[prev_doc_id]['next_doc_id'] = doc_id
                prev_doc_id = doc_id # Simply indicate it was the previous
                interval_count = (interval_count + 1) % docs_interval
            
        # Write back to new file, update dictionary
        try:
            with open(new_block_file, 'rb+') as nf:
                new_word_posting_ser = pickle.dumps(postings)

                # Store in new block file
                nf.seek(write_start)
                postings_len = nf.write(new_word_posting_ser)

                dictionary[word]['start'] = write_start
                dictionary[word]['length'] = postings_len
                write_start = write_start + postings_len
        # Workaround for sunfire
        except IOError:
            with open(new_block_file, 'wb') as nf:
                new_word_posting_ser = pickle.dumps(postings)

                # Store in new block file
                nf.seek(write_start)
                postings_len = nf.write(new_word_posting_ser)

                dictionary[word]['start'] = write_start
                dictionary[word]['length'] = postings_len
                write_start = write_start + postings_len

# Write each word, return its position to dictionary
def write_to_block(current_block, existing_block_file, dictionary, tf_weights):
    # Create new block
    if existing_block_file[0] == None and existing_block_file[1] == None:
        block_path = './block0.txt'
        start = 0
        with open(block_path, 'wb') as block_file:
            for word in dictionary:

                # Update tf_weights for postings
                for doc_id in current_block[word]:
                    current_block[word][doc_id]['tf_weight_norm'] = tf_weights[doc_id][word]

                word_docs_postings = current_block[word]
                i_word_docs_postings = {int(k): v for k, v in word_docs_postings.items()}
                sorted_word_docs_postings = dict(sorted(i_word_docs_postings.items()))
                docs_postings = pickle.dumps(sorted_word_docs_postings)
                postings_len = block_file.write(docs_postings)
                dictionary[word]['start'] = start
                dictionary[word]['length'] = postings_len
                start = start + postings_len
        existing_block_file[0] = block_path
        existing_block_file[1] = './block1.txt'
    # Merge into previous block
    # We block_split by document count, so our merging is straightforward
    # Just add in the document to the postings for each word
    else:
        prev_block_path = existing_block_file[0]
        new_block_path = existing_block_file[1]
        start = 0
        with open(new_block_path, 'wb') as block_file:
            with open(prev_block_path, 'rb') as prev_block_file:
                for word in dictionary:
                    # Word not yet written to dictionary
                    if dictionary[word]['start'] == 0 and dictionary[word]['length'] == 0:
                        
                        # Update tf_weights for postings
                        for doc_id in current_block[word]:
                            current_block[word][doc_id]['tf_weight_norm'] = tf_weights[doc_id][word]

                        word_docs_postings = current_block[word]
                        i_word_docs_postings = {int(k): v for k, v in word_docs_postings.items()}
                        sorted_word_docs_postings = dict(sorted(i_word_docs_postings.items()))
                        docs_postings = pickle.dumps(sorted_word_docs_postings)

                        postings_len = block_file.write(docs_postings)
                        dictionary[word]['start'] = start
                        dictionary[word]['length'] = postings_len
                        start = start + postings_len
                    # Word already written to dictionary
                    else:
                        # Get previous postings for current word
                        prev_word_start = dictionary[word]['start']
                        prev_word_len = dictionary[word]['length']
                        prev_block_file.seek(prev_word_start)
                        prev_word_posting_ser = prev_block_file.read(prev_word_len)
                        prev_word_posting = pickle.loads(prev_word_posting_ser)

                        # Get current postings for current word
                        cur_word_posting = {} # Not in current block
                        if current_block.get(word) != None:
                            
                            # Update tf_weights for postings
                            for doc_id in current_block[word]:
                                current_block[word][doc_id]['tf_weight_norm'] = tf_weights[doc_id][word]

                            cur_word_posting = current_block[word]

                        # Merge the postings
                        new_word_posting = {} # CLEANER but less performant
                        new_word_posting.update(prev_word_posting)
                        new_word_posting.update(cur_word_posting)

                        # Ensure sorting by doc_id
                        i_word_docs_postings = {int(k): v for k, v in new_word_posting.items()}
                        sorted_word_docs_postings = dict(sorted(i_word_docs_postings.items()))
                        new_word_posting_ser = pickle.dumps(sorted_word_docs_postings)

                        # Store in new block file
                        postings_len = block_file.write(new_word_posting_ser)
                        dictionary[word]['start'] = start
                        dictionary[word]['length'] = postings_len
                        start = start + postings_len

        # Update the prev and current block files
        tmp = existing_block_file[0]
        existing_block_file[0] = existing_block_file[1]
        existing_block_file[1] = tmp


# BOILERPLATE
def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

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
