#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import math

# Deser
import pickle


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    # Load dict_file into mem
    dictionary = pickle.load(open(dict_file, "rb"))
    compiled_results = []

    # Load queries into mem
    queries = parse_qf(queries_file)

    for query in queries:  
        search_results = find_top_search_results(query, dictionary, postings_file)
        compiled_results.append(search_results)

    # Format: 100 1000 1100 ...
    write_postings_to_file(compiled_results, results_file)


# run queries
# q := query, e.g. Hello World
# postings := { <doc_id>: {'tf_weight_norm': tf, 'next_doc_id': x, 'skip_doc_id': y}, ... }
# dictionary := {<word>: {'idf_weight': idf, 'curr_id': doc_id, 'start': x, 'length': y}, ..., }
def find_top_search_results(query, dictionary, postings_file):
    query_tf_idf_norm_weights = calc_query_tf_idf_norm_weights(dictionary, query) # Count term freq for query
    top_docs_id = []
    relevant_doc_ids = []
    query_term_postings = None
    memo_postings = {}

    with open(postings_file, 'rb') as pf:

        # Find all relevant doc ids to be compared
        for query_term in query_tf_idf_norm_weights:
            if dictionary.get(query_term) == None:
                continue
            query_term_postings = get_postings(memo_postings, dictionary, query_term, pf)
            query_term_doc_ids = list(query_term_postings.keys())
            relevant_doc_ids = list(set(relevant_doc_ids)|set(query_term_doc_ids))

        # Find cosine similarity between query and each relevant doc_id, and store each score
        for doc_id in relevant_doc_ids:
            total_cosine_score = 0

            for query_term in query_tf_idf_norm_weights:
                query_term_postings = get_postings(memo_postings, dictionary, query_term, pf)
                if (doc_id not in query_term_postings):
                    continue
                term_cosine_score = query_term_postings[doc_id]['tf_weight_norm'] * query_tf_idf_norm_weights[query_term] # Given lnc.ltc
                total_cosine_score += term_cosine_score

            top_docs_id.append((total_cosine_score, doc_id))

    if top_docs_id == []:
        return []
    else:         
        top_docs_id.sort(reverse=True) # Sort in descending order based on cosine score
        top_docs_id = top_docs_id[:10] # Take top highest scores (at most 10)
        ascending_order_for_same_relevance(top_docs_id)
        top_docs_id = [doc_id for (cosine_score, doc_id) in top_docs_id]
        return top_docs_id
        
def calc_query_tf_idf_norm_weights(dictionary, query):
    res = {}
    sum_of_squares = 0

    # Find tf for all query terms
    for query_term in query:
        if query_term not in dictionary:
            continue
        if res.get(query_term) == None:
            res[query_term] = 0
        res[query_term] += 1

    # Find tf-idf for all query terms
    for query_term in res:
        tf = res[query_term]
        idf = dictionary[query_term]['idf_weight'] # Use idf from dictionary/collection
        res[query_term] = (1 + math.log(tf, 10)) * idf # tf >= 1
        sum_of_squares += (res[query_term])**2

    # Find query length 
    query_length = math.sqrt(sum_of_squares)

    # Do normalization
    for query_term in res:
        res[query_term] /= query_length

    return res

# For doc_id with the same relevance (i.e. cosine scores), sort by increasing order
def ascending_order_for_same_relevance(top_docs):
    curr_cosine_score = top_docs[0][0]   
    index_curr = 0                    
    same_score_index_start = 0      
    
    while (index_curr < len(top_docs) - 1):
        
        if (index_curr == len(top_docs) - 2):
            if top_docs[index_curr+1][0] == curr_cosine_score:
                index_curr += 1
            top_docs[same_score_index_start:index_curr+1] = top_docs[same_score_index_start:index_curr + 1][::-1]
            break

        if (top_docs[index_curr+1][0] != curr_cosine_score):
            if index_curr == same_score_index_start:
                same_score_index_start += 1
            else:   
                top_docs[same_score_index_start:index_curr+1] = top_docs[same_score_index_start:index_curr + 1][::-1]
                same_score_index_start = index_curr + 1
            curr_cosine_score = top_docs[index_curr+1][0]

        index_curr += 1

def get_postings(memo_postings, dictionary, word, pf):
    if word in memo_postings:
        return memo_postings.get(word)

    # Not in cache
    if (dictionary.get(word) == None):
        return {}
    start = dictionary[word]['start']
    length = dictionary[word]['length']
    pf.seek(start)
    postings = pf.read(length)
    deser_postings = pickle.loads(postings)
    memo_postings[word] = deser_postings
    return deser_postings

def parse_qf(queries_file):
    result = []
    with open(queries_file, "r") as qf:
        for line in qf.readlines():
            query = tokenize(line) # [Hello, World]
            result.append(query)
    return result

def tokenize(query_sentence): # e.g. Hello world
    query_terms = nltk.word_tokenize(query_sentence)
    stemmer = nltk.stem.porter.PorterStemmer()
    stemmed_and_lowered_queries = map(lambda qt: stemmer.stem(qt.lower()), query_terms)
    return list(stemmed_and_lowered_queries)

def write_postings_to_file(compiled_results, out_file):
    with open(out_file, 'w') as f:
        for doc_id_list in compiled_results:
            results = [str(doc_id) for doc_id in doc_id_list]
            output_line = " ".join(results) + '\n'
            f.write(output_line)

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
