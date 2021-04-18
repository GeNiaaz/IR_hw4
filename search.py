import sys
import getopt
import csv
import time
import pickle
import nltk
import numpy as np
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

stemmer = Stemmer.Stemmer("english")
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, ngram_range=(1, 3))
        self.analyze = super().build_analyzer()

    def build_analyzer(self):
        return lambda doc: (stemmer.stemWord(word) for word in self.analyze(doc))

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


def run_search(dict_file, postings_file, queries_file, results_file):
    start_time = time.time()

    print("loading files from disk...")
    with open(postings_file, "rb") as matrix_file:
        matrix = pickle.load(matrix_file)
    with open(dict_file, 'rb') as dict_file:
        vectorizer = pickle.load(dict_file)
    with open('docs.txt', 'rb') as doc_file:
        docs = pickle.load(doc_file)

    print("retrieving query...")
    relevant_cols = []
    with open(queries_file) as query_file:
        content = query_file.readlines()
        query = content[0]
        for i in range(1, len(content)):
            relevant_cols.append(docs.index(content[i].strip()))

    print("finding matching documents...")
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, matrix).flatten()
    result_docs = [docs[doc] for doc in cosine_similarities.argsort()[:-11:-1]]

    print("writing to file...")
    with open(results_file, 'w') as output_file:
        write_string = ""
        doc_count = 0
        for doc in result_docs:
            doc_count += 1
            write_string += doc
            if doc_count != len(result_docs):
                write_string += " "
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
