import os
import sys
import re
import getopt
import numpy as np

def usage():
    print("usage: python3 " + sys.argv[0] + " -q file-of-queries -o output-file-of-results")


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

# Input: Str
# Output: A list of queries, these are phrasal / bag of words. Intersection is performed on their results.
def parse_query(s):
    queries = s.split(" AND ")
    return queries

def runsearch(query_file, output_file, dict_file, posting_file):

    query, list_of_relevance = parse_query_file(query_file)


if __name__ == '__main__':

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

    if file_of_queries == None or file_of_output == None:
    #if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
        usage()
        sys.exit(2)

    run_search(file_of_queries, file_of_output, dictionary_file, postings_file)
