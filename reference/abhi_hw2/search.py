#!/usr/bin/python3

"""
A0181059A
Abhijit Ravichandran
e0302456@u.nus.edu

CS3245 Homework 2
- Searching
"""

import re
import nltk
import sys
import getopt
# ------------
import pickle

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

"""
Convert boolean expression to post-fix notation for processing.
"""
def shunting_yard(query, operators):
    output = []
    op_stack = []
    for term in query:
        if term not in operators:
            # add non-operators straight to output
            output.append(term)
        elif term == "(" or term == "NOT":
            op_stack.append(term)
        elif term == ")":
            # find matching bracket
            while op_stack and op_stack[-1] != "(":
                # add everything in between brackets to output
                output.append(op_stack.pop())

            # discard the matching bracket
            if op_stack:
                op_stack.pop()
        else:
            # term is an operator
            while (op_stack and op_stack[-1] != "(" and operators.index(op_stack[-1]) <= operators.index(term)):
                output.append(op_stack.pop())
            op_stack.append(term)

    # add all remaining terms from stack to output
    while op_stack:
        output.append(op_stack.pop())

    return output

"""
Flatten NOT operations to simplify Boolean queries,
using de Morgan's law.
e.g. NOT (x AND y) == NOT x OR NOT y
"""
def flatten_nots(query):
    not_watch = False
    not_count = 0
    final_query = []

    i = 0
    while i < len(query):
        term = query[i]

        if term == "NOT" and not_watch == True:
            # more than one NOT in a row
            not_count += 1
        elif term == "NOT" and not_watch == False:
            # start looking out for NOTs
            not_watch = True
            not_count += 1
        elif term != "NOT" and not_watch:
            if not_count % 2 == 0:
                # even number of NOTs cancel out to nothing
                final_query.append(term)
            elif not_count % 2 != 0:
                # odd number of NOTs cancel out to just NOT
                if term == "(":
                    # if NOT before bracket, apply NOT to all terms in bracket
                    old_i = i
                    while (query[i] != ")"):
                        i += 1

                    # recursive call for terms within bracket
                    bracket_term = flatten_nots(query[old_i+1:i])
                    not_watch = False
                    # flatten nots within brackets
                    for subterm in bracket_term:
                        if subterm == "AND":
                            final_query.append("OR")
                        elif subterm == "OR":
                            final_query.append("AND")
                        elif subterm == "NOT":
                            not_watch = True
                        else:
                            if not_watch:
                                final_query.append(subterm)
                            else:
                                final_query.append("NOT")
                                final_query.append(subterm)

                else:
                    final_query.append("NOT")
                    final_query.append(term)

            not_watch = False
            not_count = 0

        else:
            final_query.append(term)
        i += 1
    return final_query

"""
Helps resolve NOT operations in post-fix by replacing e.g. "a NOT" to
a single term "~a".
To facilitate retrieval of ~a from postings list in searching operations.
"""
def not_to_tilde(query):
    final_query = []
    for i in range(0, len(query)):
        # if operator after term exists and is "not":
        if (i+1 < len(query) and query[i+1] == "NOT"):
            # replace with "~term"
            final_query.append("~" + query[i])
        elif query[i] == "NOT":
            # discard "not"
            continue
        else:
            final_query.append(query[i])
    return final_query

"""
Checks if a boolean query is valid before using it in search.
"""
def valid_expression_check(query):
    op_watch = False
    # temporarily remove brackets from query
    # to find consecutive ANDs or ORs
    query = list(filter(lambda char: char != "(" and char != ")", query))
    for term in query:
        if (term == "AND" or term == "OR") and not op_watch:
            # start looking for consecutive ANDs or ORs
            op_watch = True
        elif (term == "AND" or term == "OR") and op_watch:
            # consecutive ANDs or ORs found, not valid
            return False
        elif term != "AND" and term != "OR" and term != "NOT" and op_watch:
            # no consecutive ANDs or ORs yet
            op_watch = False
        else:
            continue
    # if query doesn't end on unresolved AND or OR, and query length is > 0, it's valid
    if not op_watch and len(query) > 0 and query[-1] != "NOT" and query[0] != "AND" and query[0] != "OR":
        return True
    else:
        return False

"""
Postings list merging process, for AND and OR operations.
Walks through postings lists of two operands, using skip
pointers where available.
"""
def walk_through_postings(parsed_query, operators):
    i = 0
    while i < len(parsed_query):
        # retrieve term in boolean query
        term = parsed_query[i]
        # if term is an operator (AND, OR):
        if term in operators:
            # pointers to current value in postings list
            op1_pointer = 0
            op2_pointer = 0
            # get term freq, docIds list, skip pointer list for each term involved in operation
            op1_freq, op1_ids, op1_skips = parsed_query[i-1][0], parsed_query[i-1][1], parsed_query[i-1][2]
            op2_freq, op2_ids, op2_skips = parsed_query[i-2][0], parsed_query[i-2][1], parsed_query[i-2][2]
            # store results of query here
            new_ids = []
            # if AND operation:
            if term == "AND":
                while op1_pointer < op1_freq and op2_pointer < op2_freq:
                    # compare terms at current pointers of both postings lists
                    op1_term = op1_ids[op1_pointer]
                    op2_term = op2_ids[op2_pointer]
                    # if terms are equal:
                    if op1_term == op2_term:
                        # add to result
                        new_ids.append(op1_term)
                        op1_pointer += 1
                        op2_pointer += 1
                    # if terms are unequal:
                    else:
                        # if first operand is greater and skip pointer can be used:
                        if op1_term < op2_term and op1_pointer in op1_skips and op1_skips.index(op1_pointer) != len(op1_skips) - 1 and op1_ids[op1_skips[op1_skips.index(op1_pointer) + 1]] <= op2_term:
                            op1_pointer = op1_skips[op1_skips.index(op1_pointer) + 1]
                        # if first operand is greater and no skip pointer available:
                        elif op1_term < op2_term:
                            op1_pointer += 1
                        # if second operand is greater and skip pointer can be used:
                        elif op2_term < op1_term and op2_pointer in op2_skips and op2_skips.index(op2_pointer) != len(op2_skips) - 1 and op2_ids[op2_skips[op2_skips.index(op2_pointer) + 1]] <= op1_term:
                            op2_pointer = op2_skips[op2_skips.index(op2_pointer) + 1]
                        # if second operand is greater and no skip pointer available:
                        elif op2_term < op1_term:
                            op2_pointer += 1
            # if OR operation:
            elif term == "OR":
                while op1_pointer < op1_freq and op2_pointer < op2_freq:
                    # add all docIds from both postings lists
                    op1_term = op1_ids[op1_pointer]
                    op2_term = op2_ids[op2_pointer]

                    # if first operand is smaller; catch up to second, adding all along the way
                    if op1_term < op2_term:
                        new_ids.append(op1_term)
                        op1_pointer += 1
                    # if second operand is smaller; catch up to first, adding all along the way
                    elif op2_term < op1_term:
                        new_ids.append(op2_term)
                        op2_pointer += 1
                    # if both equal, add both, advance both pointers
                    elif op1_term == op2_term:
                        new_ids.append(op1_term)
                        op1_pointer += 1
                        op2_pointer += 1

                # if both lists are unequal size, add remaining items from longer list
                if op1_pointer < op1_freq:
                    new_ids.extend(op1_ids[op1_pointer:])
                elif op2_pointer < op2_freq:
                    new_ids.extend(op2_ids[op2_pointer:])

            # two operands and an operator have been resolved; delete them
            del parsed_query[i-2:i+1]
            i -= 1
            # replace with documents that are result of query
            parsed_query.insert(i-1, [len(new_ids), new_ids, []])
        else:
            i += 1

    return parsed_query[0][1]

"""
Main function to read queries from file, pre-process queries,
and execute search function.
Using the given dictionary file and postings file,
perform searching on the given queries file and output the results to a file
"""
def run_search(dict_file, postings_file, queries_file, results_file):
    print('running search on the queries...')

    # allowed operators
    operators = ['(', ')', 'NOT', 'AND', 'OR']
    Porter = nltk.stem.PorterStemmer()

    # read queries from query file
    queries = []
    with open(queries_file) as file:
        for line in file:
            query = []
            # tokenize each word in query
            for word in nltk.word_tokenize(line):
                # if token is not an operator:
                if word not in operators:
                    # stem non-operator words in query
                    query.append(Porter.stem(word.lower()))
                else:
                    query.append(word)
            queries.append(query)
    file.close()

    # load dictionary with pointers to token data to memory
    with open(dict_file, "rb") as dictionary_file:
        dictionary = pickle.load(dictionary_file)
    dictionary_file.close()

    # final results for each query stored here
    search_results = []
    for query in queries:
        # if query is not valid, return empty line for query
        if not valid_expression_check(query):
            search_results.append([])
            print("invalid query: " + "'" + ' '.join(query) + "'")
            continue

        # pre-process query
        parsed_query = not_to_tilde(shunting_yard(flatten_nots(query), operators))

        with open(postings_file, "rb") as post_file:
            # for each term in query:
            for i in range(0, len(parsed_query)):
                term = parsed_query[i]
                # if term is not an operator:
                if term not in operators:
                    try:
                        # retrieve postings list data for that term, using pointer from dict
                        post_file.seek(dictionary[term][1])
                        term_freq = dictionary[term][0]
                        term_name, posting_list, skips = pickle.loads(post_file.read(dictionary[term][2]))
                        # replace term in query with its postings and skip data from postings file
                        parsed_query[i] = [term_freq, posting_list, skips]
                    except KeyError:
                        # if term not in corpus / postings, replace with empty data
                        parsed_query[i] = [0, [], []]
        post_file.close()

        # call function to carry out merging of postings lists, and add results to list
        search_results.append(walk_through_postings(parsed_query, operators))

    # write results to output file
    with open(results_file, 'w') as output_file:
        for i in search_results:
            i.sort()
            write_string = ""
            for j in i:
                write_string += (str(j) + " ")
            output_file.write(write_string + "\n")

    print("\ndone.")
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
