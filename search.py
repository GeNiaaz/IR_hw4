import sys
import time
import getopt
import csv
import gzip
import pickle
import nltk
from nltk.corpus import wordnet
import numpy as np
from scipy import mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

query_refinement = False
query_synonym_expansion = True

stemmer = nltk.stem.snowball.SnowballStemmer("english")
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, dtype=np.float32)
        self.analyze = super().build_analyzer()

    def build_analyzer(self):
        return lambda doc: (stemmer.stem(word) for word in self.analyze(doc))


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def find_docs_for_phrasal_query(query, vectorizer, dictionary):
    # Assume query comes with the quotation marks, i.e. => "hello there world"
    query_words = []
    for word in vectorizer.build_analyzer()(query):
        query_words.append(word)

    # Find docs containing all query words
    position_array = [dictionary[word] for word in query_words]
    common_docs = set.intersection(*[set(array.keys()) for array in position_array])

    final_result = set()
    for doc in common_docs:
        positions = [array[doc] for array in position_array]
        for pos in positions[0]:
            exist_check = [pos+i in positions[i] for i in range(1, len(positions))]
            if False not in exist_check:
                final_result.add(doc)

    return final_result


def process_query(query, vectorizer, positions):
    queries = query.split(" AND ") # look for ANDs if any
    phrases_found = False
    phrases_doc_set = set()
    phrase_count = 0
    non_phrasal_query = ""

    for i in range(len(queries)):
        query = queries[i]
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            # phrasal query
            phrases_found = True
            query = query[1:-1]
            phrasal_docs = find_docs_for_phrasal_query(query, vectorizer, positions)
            if phrase_count == 0:
                phrases_doc_set = phrasal_docs
            else:
                phrases_doc_set = set.intersection(phrases_doc_set, phrasal_docs)
            phrase_count += 1
        else:
            if query_synonym_expansion:
                query = query_synonym_extension(query)
            non_phrasal_query += query
            if i != len(queries) - 1:
                non_phrasal_query += " "
    return (non_phrasal_query, list(phrases_doc_set), phrases_found)

# alpha beta are parameters to be adjusted
# varargs are vectors of docs of inputs from query
def rocchio_calculation(alpha, beta, query_vec, doc_vecs):
    print("refining query with Rocchio algorithm...")
    weighted_query_np = query_vec * alpha
    mean_np = np.mean(doc_vecs, axis=0)
    weighted_mean_np = mean_np * beta
    final_result = weighted_query_np + weighted_mean_np
    return final_result


# Adding synonyms to expand query
# Capped to 1 synonym per word to minimise leading the query
# 2xOriginal terms to ensure original terms have higher weightage than synonyms
def query_synonym_extension(query):
    print("expanding query with synonyms...")
    list_of_original_terms = nltk.word_tokenize(query)
    new_query_list = []

    for term in list_of_original_terms:
        new_query_list.append(term)
        one_term_cap = False
        for syn in wordnet.synsets(term):
            if one_term_cap:
                break
            for lemm in syn.lemma_names():
                if lemm not in list_of_original_terms:
                    new_query_list.append(lemm)
                    one_term_cap = True
                    break

    new_query = " ".join(term for term in new_query_list).strip()
    return new_query


def write_to_file(result_docs, results_file):
    print(len(result_docs), "matching documents found.")
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


def run_search(dict_file, postings_file, queries_file, results_file):
    start_time = time.time()

    print("loading files from disk...")
    with gzip.open(postings_file, 'rb') as post_file:
        matrix = pickle.load(post_file)
    with gzip.open(dict_file, 'rb') as dict_file:
        vectorizer = pickle.load(dict_file)
    with gzip.open('docs.txt', 'rb') as doc_file:
        docs = pickle.load(doc_file)
    with gzip.open('positions.txt', 'rb') as posn_file:
        positions = pickle.load(posn_file)

    print("processing query...")
    relevant_cols = []
    with open(queries_file) as query_file:
        content = query_file.readlines()
        unprocessed_query = content[0].strip()
        (query, phrases_doc_list, phrases_found) = process_query(unprocessed_query, vectorizer, positions)
        for i in range(1, len(content)):
            relevant_cols.append(docs.index(content[i].strip()))

    print("finding matching documents...")
    if query == "":
        for doc in phrases_doc_list:
            if doc not in relevant_cols:
                relevant_cols.append(doc)
        write_to_file(relevant_cols, results_file)
    else:
        query_vector = vectorizer.transform([query])
        if query_refinement and len(relevant_cols) != 0:
            relevant_vectors = matrix[relevant_cols, :]
            refined_vector = rocchio_calculation(0.7, 0.3, query_vector, relevant_vectors)
        else:
            refined_vector = query_vector

        cosine_similarities = cosine_similarity(refined_vector, matrix).flatten()
        sorted_cosines = cosine_similarities.argsort()[::-1]
        result_docs = [docs[doc] for doc in sorted_cosines if cosine_similarities[doc] != 0]

        if phrases_found: # i.e. phrases were found, intersect results w/ phrase doc set
            print('test')
            result_docs = [doc for doc in result_docs if doc in phrases_doc_list]

        write_to_file(result_docs, results_file)

    end_time = time.time()
    print("done.")
    print(results_file, "generated.")
    print("search completed in", round(end_time - start_time, 5), "secs.")
    #print(matrix.shape)

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
