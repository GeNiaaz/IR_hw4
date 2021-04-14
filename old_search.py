import os
import sys
import re
import getopt
from whoosh import index
from whoosh.qparser import QueryParser

def usage():
    print("usage: python3 " + sys.argv[0] + " -q file-of-queries -o output-file-of-results")

def run_search(query_file, output_file):
    print("Searching documents...")
    relevance_judgments = []
    with open(query_file, encoding="utf-8", errors="ignore", mode="r") as file:
        lines = file.read().splitlines()
        query = lines[0]
        for i in range(1, len(lines)):
            relevance_judgments.append(lines[i])

    result_array = []
    ix = index.open_dir("index-dir")
    with ix.searcher() as searcher:
        parser = QueryParser("content", schema=ix.schema)
        parsed_query = parser.parse(query)
        results = searcher.search(parsed_query, limit=None)
        for hit in results:
            result_array.append(hit["document_id"])

    with open(output_file, 'w') as results_file:
        write_string = ""
        value_count = 0
        for value in result_array:
            write_string += (value)
            value_count += 1
            if value_count != len(result_array):
                write_string += " "
        results_file.write(write_string)

    print("Done.")
    print("Results written to", output_file + ".")

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

run_search(file_of_queries, file_of_output)
