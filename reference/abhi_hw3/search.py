#!/usr/bin/python3

"""
A0181059A
Abhijit Ravichandran
e0302456@u.nus.edu

CS3245 Homework 3
"""

import re
import nltk
import sys
import getopt
# ------------
import heapq
import string
import pickle
import math

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    Porter = nltk.stem.PorterStemmer()
    """
    filter out all punctuation except for hyphens, forward slashes, dots and commas,
    as they are necessary for numbers, dates and phrases
    """
    reg_exp = " |;|&|<|>|\(|\)|\[|\]|\{|\}|\"|\n|\'|\.\.\.|\+|\:|\;|\!|\?| $| #|% "
    # dict to cache retrieved word data from pointers
    word_data = {}

    # unpickle and load dictionary from file
    with open(dict_file, "rb") as dictionary_file:
        dictionary = pickle.load(dictionary_file)
    dictionary_file.close()

    # open query file, postings and results file
    with open(queries_file) as query_file, open(postings_file, "rb") as post_file, open(results_file, "w") as output_file:
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
                    scores[doc] += (query[word] / query_total) * postings[doc]

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

    print("done.")
    print(results_file, "generated.")

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
