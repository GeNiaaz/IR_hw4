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

# query refinement (Rocchio) and synonym expansion options
# can be enabled or disabled as per your wish
# query synonyms left on as we found it helps with accuracy
query_refinement = False
query_synonym_expansion = True

# Snowball stemmer for more accurate stemming
stemmer = nltk.stem.snowball.SnowballStemmer("english")

# modified version of TfidfVectorizer that converts list of raw text to a term-doc matrix
# modified to use float32 to save space when saved to disk, and to stem all words automatically
# used here to convert query to a vector
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, dtype=np.float32)
        self.analyze = super().build_analyzer()

    # function that processes all text passed to vectorizer when called
    def build_analyzer(self):
        # automatically stem all words in text passed to it
        return lambda doc: (stemmer.stem(word) for word in self.analyze(doc))

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

# find phrases in query, as denoted by quotation marks
def find_docs_for_phrasal_query(query, vectorizer, dictionary):
    # stem and case-fold terms in query
    query_words = []
    for word in vectorizer.build_analyzer()(query):
        query_words.append(word)

    # get dictionary containing positions of every word in phrase, for every document it appears in
    position_array = [dictionary[word] for word in query_words]
    # use set intersections to efficiently obtain common docs all these words appear in
    common_docs = set.intersection(*[set(array.keys()) for array in position_array])

    final_result = set()
    # for each common document:
    for doc in common_docs:
        # obtain the positions of each word in the phrase, in that document
        positions = [array[doc] for array in position_array]
        for pos in positions[0]:
            # check if the words appear in the right phrasal order in that document
            exist_check = [pos+i in positions[i] for i in range(1, len(positions))]
            # if they do (i.e. not False), add document to set
            # use set to prevent duplicates
            if False not in exist_check:
                final_result.add(doc)

    return final_result

# main query processing function
# calls phrasal query processing function, and query synonym expansion function (if activated)
def process_query(query, vectorizer, positions):
    # split query up by ANDs, if any
    queries = query.split(" AND ")
    phrases_found = False
    phrases_doc_set = set()
    phrase_count = 0
    non_phrasal_query = ""

    # for each term in query:
    for i in range(len(queries)):
        query = queries[i]
        # if phrase found:
        if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
            # phrasal query
            phrases_found = True
            query = query[1:-1]
            # find documents that match phrase
            phrasal_docs = find_docs_for_phrasal_query(query, vectorizer, positions)
            # if multiple phrases in query, they must be ANDed together; intersect their results
            if phrase_count == 0:
                phrases_doc_set = phrasal_docs
            else:
                phrases_doc_set = set.intersection(phrases_doc_set, phrasal_docs)
            query = query.split(" ")
            for j in range(len(query)):
                term = query[j]
                if query_synonym_expansion:
                    term = query_synonym_extension(term, vectorizer)
                non_phrasal_query += term
                if not (i == len(queries) - 1 and j == len(query) - 1):
                    non_phrasal_query += " "
            phrase_count += 1
        else:
            # free text term in query
            # if synonym expansion enabled; add synonyms of this term to query
            if query_synonym_expansion:
                query = query_synonym_extension(query, vectorizer)
            non_phrasal_query += query
            # add space if not final term in query
            if i != len(queries) - 1:
                non_phrasal_query += " "
    return (non_phrasal_query, list(phrases_doc_set), phrases_found)

# alpha beta are parameters to be adjusted
# varargs are vectors of docs of inputs from query
def rocchio_calculation(alpha, beta, query_vec, doc_vecs):
    print("refining query with Rocchio algorithm...")
    # multiply query vector by weight
    weighted_query_np = query_vec * alpha
    # obtain centroid of document vectors known to be relevant
    mean_np = np.mean(doc_vecs, axis=0)
    # multiply by weight
    weighted_mean_np = mean_np * beta
    # add weighted centroid and query vector together to obtain refined vector
    final_result = weighted_query_np + weighted_mean_np
    return final_result

# Adding synonyms to expand query
# Capped to 1 synonym per word to minimise leading the query
# 2xOriginal terms to ensure original terms have higher weightage than synonyms
def query_synonym_extension(query, vectorizer):
    # split query up into tokens
    list_of_original_terms = nltk.word_tokenize(query)
    new_query_list = []

    # for each term in query:
    for term in list_of_original_terms:
        # obtain synonyms
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

    # join list back up into string
    new_query = " ".join(term for term in new_query_list).strip()
    return new_query

# given list of docs judged by program to be relevant,
# write docs to output file
def write_to_file(result_docs, results_file):
    print(len(result_docs), "matching documents found.")
    print("writing to file...")

    with open(results_file, 'w') as output_file:
        write_string = ""
        doc_count = 0
        for doc in result_docs:
            doc_count += 1
            write_string += doc
            # if not last document in array: add space
            if doc_count != len(result_docs):
                write_string += " "
        output_file.write(write_string)

# main function to run search, that calls query pre-processing functions,
# query vectorizing functions, query refinement functions, and calculates cosine
# similarities to obtain relevant document vectors.
def run_search(dict_file, postings_file, queries_file, results_file):
    # time search
    start_time = time.time()

    # load term-doc vector matrix, vectorizer, matrix column - termId mapping, word positions index
    print("loading files from disk...")
    with gzip.open(postings_file, 'rb') as post_file:
        matrix = pickle.load(post_file)
    with gzip.open(dict_file, 'rb') as dict_file:
        vectorizer = pickle.load(dict_file)
    with gzip.open('docs.txt', 'rb') as doc_file:
        docs = pickle.load(doc_file)
    with gzip.open('positions.txt', 'rb') as posn_file:
        positions = pickle.load(posn_file)

    # retrieve query and relevance judgments from file
    print("processing query...")
    relevant_cols = []
    with open(queries_file) as query_file:
        content = query_file.readlines()
        unprocessed_query = content[0].strip()
        # process phrases and boolean elements of query, return documents that match phrase (if any)
        (query, phrases_doc_list, phrases_found) = process_query(unprocessed_query, vectorizer, positions)
        # retrieve relevance judgments
        for i in range(1, len(content)):
            relevant_cols.append(docs.index(content[i].strip()))

    print("finding matching documents...")
    # if query comprised entirely of phrases: just return documents from query processing function earlier
    if query == "":
        for doc in phrases_doc_list:
            if doc not in relevant_cols:
                relevant_cols.append(doc)
        write_to_file(relevant_cols, results_file)
    # if free-text elements also exist: convert to vector
    else:
        # vectorize query
        query_vector = vectorizer.transform([query])
        # if enabled, use Rocchio algorithm to optimize query vector
        if query_refinement and len(relevant_cols) != 0:
            relevant_vectors = matrix[relevant_cols, :]
            refined_vector = rocchio_calculation(0.7, 0.3, query_vector, relevant_vectors)
        # else, use query vector to calculate cosine scores straightaway
        else:
            refined_vector = query_vector

        # calculate cosine distance of each document vector in matrix with query vector
        cosine_similarities = cosine_similarity(refined_vector, matrix).flatten()
        # sort document vectors in decreasing order of cosine similarity from query vector
        sorted_cosines = cosine_similarities.argsort()[::-1]
        # return all vectors with non-zero cosine similarity, in that decreasing order
        result_docs = [docs[doc] for doc in sorted_cosines if cosine_similarities[doc] != 0]

        if phrases_found: # i.e. phrases were found, intersect results w/ phrase document set
            result_docs.extend([doc for doc in phrases_doc_list if doc not in result_docs])

        write_to_file(result_docs, results_file)

    # stop timer
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
