#!/usr/bin/python3

import re
import nltk
import sys
import getopt
# ------------
import heapq
import gzip
import string
import pickle
import math
import time
import numpy as np

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

# alpha beta are args to be adjusted
# varargs are vectors of docs of inputs from query
def rocchio_calculation(alpha, beta, query_vec, *doc_vecs):
    weighted_query_np = np.multiply(alpha, query_vec)
    weighted_query_list = weighted_query_np.tolist()

    mean_np = np.mean(doc_vecs, axis=0)
    weighted_mean_np = np.multiply(beta, mean_np)
    weighted_mean_list = weighted_mean_np.tolist()

    final_result = np.sum((weighted_query_list, weighted_mean_list), axis=0)
    final_result_list = final_result.tolist()
    return final_result_list


def parse_query_file(query_f):
    # Vars to return
    query = ""
    list_of_relevance = []

    # Parsing query file
    with open(query_f, 'r') as query_readable:
        raw_input = []
        for line in query_readable:
            raw_input.append(line.strip())

        query = parse_query(raw_input[0])
        for a in range(1, len(raw_input)):
            list_of_relevance.append(raw_input[a])

    return query, list_of_relevance


def parse_query(s):
    queries = s.split(" AND ")
    return queries

# '"i am a phrase"' (note the additional single quotes surrounding the string.)
def is_phrasal(s):
    return len(s) > 1 and s[0] == '"'

# "hello world"
def is_bog(s):
    return not is_phrasal(s)

# "hello world" -> ["hello", "world"]
def words_bog(bog):
    return bog.split()

def process_AND(list_a, list_b):
    ptr_a = 0
    ptr_b = 0

    max_index_a = len(list_a)
    max_index_b = len(list_b)

    resultant_list = []

    while ptr_a < max_index_a and ptr_b < max_index_b:
        curr_a = list_a[ptr_a]
        curr_b = list_b[ptr_b]

        if curr_a == curr_b:
            ptr_a += 1
            ptr_b += 1

            resultant_list.append(curr_a)
        else:
            if curr_a < curr_b:
                ptr_a += 1
            else:
                ptr_b += 1

    return resultant_list

postings_cache = {} # Consider caching our pickle readings for postings 

def find_docs_for_phrasal_query(query, dictionary):
    # Assume query comes with the quotation marks, i.e. => "hello there world" 
    query_words = query.split()
    
    # Find docs containing all query words
    idf_query_words = [(dictionary[word][0], word) for word in query_words]
    idf_query_words = set(idf_query_words) # remove duplicates   
    sorted_idf_query_words = sorted(idf_query_words, reverse=True)   # sort query_words by highest idf (appears in least docs)

    merged_docs = set()

    for (idf, word) in sorted_idf_query_words:
        if word not in postings_cache: # cache operation
            # post_file.seek(dictionary[word][1])
            # postings_cache[word] = pickle.loads(post_file.read(dictionary[word][2]))
            None
        else:
            if (merged_docs == set()): # set merged_docs to the first set of doc_ids
                # merged_docs = set(postings_cache[word].keys())
                None
            else: # find intersection of doc_ids sets, in increasing order of size (since sorted by highest idf) (faster merging)
                # merged_docs = merged_docs.intersection(set(postings_cache[word].keys()))
                None

    # Find docs with the exact phrase 
    result_docs = []

    for doc_id in merged_docs:
        merged_postitions = set()
        for (idx, word) in enumerate(query_words):
            positions = [(position - idx) for position in postings_cache[word][doc_id][1]] # Adjusting the positional index to that of the first query word
            if (merged_postitions == set()):
                merged_postitions = set(positions)
            else:
                merged_postitions = merged_postitions.intersection(set(positions))
        
        if merged_postitions != set():  # Consider adding weights too => larger set == more frequently-appearing phrase == more relevant
            result_docs.append(doc_id) 

    return result_docs

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    start_time = time.time()
    Porter = nltk.stem.PorterStemmer()
    """
    filter out all punctuation except for hyphens, forward slashes, dots and commas,
    as they are necessary for numbers, dates and phrases
    """
    reg_exp = " |;|&|<|>|\(|\)|\[|\]|\{|\}|\"|\n|\'|\.\.\.|\+|\:|\;|\!|\?| $| #|% "
    # dict to cache retrieved word data from pointers
    word_data = {}

    print('loading files from disk...')
    # unpickle and load dictionary from file
    with gzip.open(dict_file, "rb") as dictionary_file:
        dictionary = pickle.load(dictionary_file)
    dictionary_file.close()

    print('finding matching documents...')
    # open query file, postings and results file
    with open(queries_file) as query_file, gzip.open(postings_file, "rb") as post_file, open(results_file, "w") as output_file:
        line_count = 0
        # for each query:
        for line in query_file:
            # instantiate dict to store tf and idf for each term in query
            query = {}
            # initialize dict to keep score for each document
            scores = {}

            # for each word in query:
            for word in re.split(reg_exp, line):
                # strip starting and ending punctuation marks e.g. quotes, full stops, ellipses from each word
                if not all(letter in string.punctuation for letter in word):
                    while word[-1] in string.punctuation:
                        word = word[:-1]
                    while word[0] in string.punctuation:
                        word = word[1:]
                    # case-fold and stem word
                    word = Porter.stem(word.lower())
                    # add word to query data dict, with term freq. of 0, if not yet seen in query
                    if word not in query.keys():
                        query[word] = 0
                    # increment term freq. of word in query
                    query[word] += 1

                    """
                    retrieve postings lists for each word in query, using pointers stored in dictionary,
                    if word has not already been seen before, and store in word_data
                    """
                    if word not in word_data.keys():
                        if word in dictionary:
                            post_file.seek(dictionary[word][1])
                            idf = dictionary[word][0]
                            postings = pickle.loads(post_file.read(dictionary[word][2]))
                            # store postings and idf value for word in word_data
                            word_data[word] = [postings, idf]
                        else:
                            # if word is not in doc. collection, initialize empty postings list, and idf val. of 0
                            word_data[word] = [{}, 0]

                    idf = word_data[word][1]
                    # calculate non-normalized score for word in query
                    query[word] = idf * (1 + math.log(query[word], 10))

            # after iterating over query, compute total query vector length, to be used in normalization
            query_total = math.sqrt(sum(value ** 2 for value in query.values()))

            # for each word in collected query data:
            for word in query:
                postings = word_data[word][0]
                # for each doc in word's postings list:
                for doc in postings:
                    # initialize document in score vector if not yet seen, with score of 0
                    if doc not in scores.keys():
                        scores[doc] = 0
                    """
                    multiply document score for this word (pre-computed and normalized during indexing)
                    with query score for this word (which is divided by total query vector length for normalization),
                    to get total weight for this word. add this weight to the score for this document.
                    """
                    scores[doc] += (query[word] / query_total) * postings[doc]["tf"]

            """
            after query is iterated over entirely, construct min heap out of score dictionary,
            with -score as the primary node value. Effectively a max heap, but making all scores
            negative allows us to select documents by smallest docId first as a secondary
            deciding value, if two nodes have the same score.
            """
            heap = [(-value, key) for key, value in scores.items()]

            with open(results_file, 'a') as output_file:
                write_string = ""
                value_count = 0
                # select max. 10 docs with the 'smallest' (i.e. most negative) scores
                # if two docs have the same score, doc. with smaller docId value will be selected first
                for value in heapq.nsmallest(10, heap):
                    # write to output file
                    write_string += (str(value[1]))
                    value_count += 1
                    # insert space if value is not last in heap
                    if value_count != min(len(heap), 10):
                        write_string += " "
                line_count += 1
                # insert newline before writing line to file, if line is not the first
                if line_count != 1:
                    output_file.write("\n")
                output_file.write(write_string)

    end_time = time.time()
    print("done.")
    print(results_file, "generated.")
    print("search completed in", round(end_time - start_time, 5), "secs.")

# This is to allow us to export search.py as a library
# Otherwise, whenever `search.py` is exported, we will execute the following lines of code.
# See `search-test.py` for import use.
if __name__ == "main":
    dictionary_file = postings_file = file_of_queries = output_file_of_results = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
        usage()
        sys.exit(2)

    run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
