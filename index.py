import sys
import getopt
import csv
import gzip
import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Snowball stemmer for more accurate stemming
stemmer = nltk.stem.snowball.SnowballStemmer("english")

# modified version of TfidfVectorizer that converts list of text to a term-doc matrix
# modified to use float32 to save space when saved to disk, and to stem all words automatically
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, dtype=np.float32)
        self.analyze = super().build_analyzer()

    # function that processes all text passed to vectorizer when called
    def build_analyzer(self):
        # automatically stem all words in text passed to it
        return lambda doc: (stemmer.stem(word) for word in self.analyze(doc))

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

# main function to iterate through dataset, collect terms from case title, court and text fields
# collect positions of these terms in dictionary, fit terms to document-term vector space matrix
def build_index(in_dir, out_dict, out_postings):
    # one last check before overwriting indices
    warning_check = input("are you sure you want to overwrite your index? y/n ")
    if warning_check != "y":
        print("closing...")
        sys.exit(2)

    # to store text from each document
    corpus = []
    # to store docId - row number mapping in matrix later
    docs = []
    # store positions for every document each word is in
    positions = {}
    # case-fold and stem words; learn vocabulary to be fitted to matrix later
    vectorizer = TfidfStemVectorizer()

    # read dataset
    print("processing dataset...")
    with open(in_dir) as dataset:
        csv.field_size_limit(sys.maxsize)
        dataset_reader = csv.reader(dataset, delimiter=",")
        column_no = 1
        next(dataset) # skip header line
        for line in dataset_reader:
            print("\treading line", column_no); column_no += 1
            doc_id = line[0]
            title = line[1]
            date = line[3] # discard; dates are most likely included in text already
            court = line[4]
            text = title + " " + court + " " + line[2] # add title and court to main text

            # avoid possibility of duplicate document IDs in dataset
            if doc_id not in docs:
                # count position(s) of each word in text
                position_count = 0
                # automatically stems and case-folds each word
                for word in vectorizer.build_analyzer()(text):
                    # store word and positions in respective dictionary
                    if word not in positions:
                        positions[word] = {}
                    if doc_id not in positions[word]:
                        positions[word][doc_id] = []
                    positions[word][doc_id].append(position_count)
                    position_count += 1

                # save docId to list; order of reading docIds represents order of docIds in matrix
                docs.append(doc_id)
                # save text; to be converted to term-doc matrix
                corpus.append(text)
        dataset.close()

    print("building vector space matrix...")
    # construct term-document vector space matrix
    # each document is a row, each unique term is a column
    matrix = (vectorizer.fit_transform(corpus))

    # save structures to pickle
    print("saving to disk...")
    with gzip.open(out_postings, 'wb') as post_file:
        pickle.dump(matrix, post_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(out_dict, 'wb') as dict_file:
        pickle.dump(vectorizer, dict_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open('docs.txt', 'wb') as doc_file:
        pickle.dump(docs, doc_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open('positions.txt', 'wb') as posn_file:
        pickle.dump(positions, posn_file, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
