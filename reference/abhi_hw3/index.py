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
import os
import string
import pickle
import math

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

"""
Function to iterate through each file in a folder in ascending file-name order.
"""
def filename_sort(filename):
    number_regex = re.compile(r'(\d+)')
    parts = number_regex.split(filename)
    parts[1::2] = map(int, parts[1::2])
    return parts

"""
build index from documents stored in the input directory,
then output the dictionary file and postings file
"""
def build_index(in_dir, out_dict, out_postings):
    print('indexing...')

    Porter = nltk.stem.PorterStemmer()
    """
    filter out all punctuation except for hyphens, forward slashes, dots and commas,
    as they are necessary for numbers, dates and phrases
    """
    reg_exp = " |;|&|<|>|\(|\)|\[|\]|\{|\}|\"|\n|\'|\.\.\.|\+|\:|\;|\!|\?| $| #|% "
    # dictionary to store term frequency and postings data
    word_data = {}
    # dictionary to collect document lengths
    documents = {}

    # iterate through and open each file in input dir., in ascending order
    for filename in (files for files in sorted(os.listdir(in_dir), key=filename_sort) if files != ".DS_Store"):
        with open(in_dir + "/" + os.fsdecode(filename), encoding="utf-8", errors="ignore", mode="r") as file:
            int_filename = int(filename)
            # instantiate document in documents dict with a length of 0
            documents[int_filename] = 0
            # read all text from file
            file_content = file.read()

            # filter text using regex, and iterate through each word in text
            for word in re.split(reg_exp, file_content):
                # ignore word if it only consists of punctuation marks e.g. "..." or "--"
                if not all(letter in string.punctuation for letter in word):
                    # since ending full-stops and commas are not sieved out by our regex, we remove them here
                    while word[-1] in string.punctuation:
                        word = word[:-1]
                    # case-fold and stem each token
                    word = Porter.stem(word.lower())
                    # instantiate word in word_data if word is not already seen before
                    if word not in word_data.keys():
                        # each word in word_data is linked to a postings dict with docId as key and term freq as value
                        word_data[word] = {}

                    posting = word_data[word]
                    # add current document to current word's postings if not yet added
                    if int_filename not in posting.keys():
                        posting[int_filename] = 0
                    # increment term frequency for this document, for this word
                    posting[int_filename] += 1

    # delete old dictionary and postings files if they still exist
    for file in [out_dict, out_postings]:
        if os.path.exists(file):
            os.remove(file)

    # instantiate dictionary which will be used to store terms and pointers to their postings lists later
    dictionary = {}
    # get doc. collection size, to compute idf
    collection_size = len(documents)

    # for each word seen in doc. collection
    for word in word_data.keys():
        posting = word_data[word]
        # replace the tf values for each doc. the word is seen in, with (1 + log(tf)) values
        for doc in posting.keys():
            posting[doc] = 1 + math.log(posting[doc], 10)
            # add square of (1 + log(tf)) value to that document's length in dictionary
            # to be used in normalization later
            documents[doc] += posting[doc] ** 2

    with open(out_postings, 'ab') as postings_file:
        # for each word in word_data
        for word in word_data.keys():
            posting = word_data[word]
            # for each document in word's posting list
            for doc in posting.keys():
                # normalize (1 + log(tf)) value for each doc by dividing with document length collected earlier
                posting[doc] /= math.sqrt(documents[doc])

            # calculate idf, and store idf and pointer to term data for each term in the dictionary
            dictionary[word] = [math.log(collection_size / len(posting.keys()), 10), postings_file.tell()]
            # pickle and dump term data to postings file
            postings_file.write(pickle.dumps(posting))
            # store how many bytes of data to read from initial pointer value
            dictionary[word].append(postings_file.tell() - dictionary[word][1])

    # dump dictionary to dictionary file
    with open(out_dict, 'wb') as dictionary_file:
        pickle.dump(dictionary, dictionary_file)

    print("done.")
    print(out_dict, "and", out_postings, "generated.")



input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
