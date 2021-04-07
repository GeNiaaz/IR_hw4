#!/usr/bin/python3
import string

import nltk
from nltk.stem import PorterStemmer
import sys
import math
import getopt
import os
import pickle


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')
    # Pls implement your code in below

    # Get list of doc ID's, sorted by index
    list_of_document_id = os.listdir(in_dir)
    list_of_document_id = sorted([int(doc) for doc in list_of_document_id])

    # Variables to change
    case_fold_status = True
    stem_status = True

    # Init pickle files
    final_dict_pickle_file = open(out_dict, 'ab')
    length_pickle_file = open("length.txt", 'ab')
    normalise_n_pickle_file = open(out_postings, 'ab')

    term_counter = 0
    normalise_n_index_txt = 0
    normalise_n_index_pickle = 0

    ps = PorterStemmer()

    dict_of_terms = {}
    temp_dict = {}
    postings_dict = {}

    punctuation = [",", "\"", "/", "(", ")", "/", "?", "!", "@", "#", "^", "*", "|", "+", "-", "_", "="]

    for current_doc_id in list_of_document_id:
        doc_path = in_dir + str(current_doc_id)
        file = open(doc_path, 'r')
        doc = file.read()

        print(current_doc_id)

        # Tokenizing and processing of terms in doc
        sentences = nltk.sent_tokenize(doc)
        for sentence in sentences:
            terms = nltk.word_tokenize(sentence)
            for term in terms:
                if term in string.punctuation:
                    continue
                for p in punctuation:
                    term.replace(p, "")
                if case_fold_status:
                    term = term.lower()
                if stem_status:
                    term = ps.stem(term)
                term_counter += 1

                # Updating main dictionary of terms
                if term in dict_of_terms:
                    dict_term_pairs = dict_of_terms[term]

                    # matched_pair_list = [pair for pair in dict_term_pairs if pair[0] == current_doc_id]

                    if current_doc_id in dict_term_pairs:
                        term_freq = dict_term_pairs[current_doc_id]
                        term_freq += 1

                        dict_term_pairs[current_doc_id] = term_freq

                    else:
                        dict_term_pairs[current_doc_id] = 1

                else:
                    dict_of_terms[term] = {current_doc_id: 1}

                # Updating temp dict for Length[N]
                if term in temp_dict:
                    temp_dict[term] = temp_dict[term] + 1

                else:
                    temp_dict[term] = 1

        # Writing length txt file
        doc_and_counter = (current_doc_id, term_counter)

        # Writing pickle length file
        pickle.dump(doc_and_counter, length_pickle_file)

        # Writing Length[N] file
        list_of_tf = []
        list_of_normalised_tf = []
        sum_for_normalizing = 0
        num_unique_terms = len(temp_dict)
        for key, value in temp_dict.items():
            value_to_add = 1 + math.log(value, 10)
            list_of_tf.append((key, value_to_add))

        for pair in list_of_tf:
            sum_for_normalizing += pair[1] ** 2

        normalizing_factor = sum_for_normalizing ** 0.5

        for p in list_of_tf:
            temp = p[1]
            norm_result = temp / normalizing_factor

            list_of_normalised_tf.append((p[0], norm_result))

        # Writing to pickle files
        postings_dict[current_doc_id] = list_of_normalised_tf

        temp_dict.clear()
        term_counter = 0

    final_posting_dict = {}
    for k, v in postings_dict.items():
        temp_dict = {}
        for pair in v:
            temp_dict[pair[0]] = pair[1]
        final_posting_dict[k] = temp_dict

    pickle.dump(final_posting_dict, normalise_n_pickle_file)

    dict_of_terms = dict(sorted(dict_of_terms.items()))

    print("printing to file...")
    final_dict_index_txt = 0
    final_dict_index_pickle = 0
    for key, value in dict_of_terms.items():
        list_of_postings = list(value.items())
        list_of_postings_str = str(list(value.items()))
        doc_freq = len(list_of_postings)

        # Writing to pkl for dictionary
        pickle.dump([key, doc_freq, final_dict_index_pickle], final_dict_pickle_file)


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
