import sys
import getopt
import csv
import gzip
import pickle
import joblib
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = nltk.stem.snowball.SnowballStemmer("english")
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, dtype=np.float32)
        self.analyze = super().build_analyzer()
    def build_analyzer(self):
        return lambda doc: (stemmer.stem(word) for word in self.analyze(doc))

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    warning_check = input("are you sure you want to overwrite your index? y/n ")
    if warning_check != "y":
        print("closing...")
        sys.exit(2)

    corpus = []
    docs = []
    positions = {}
    vectorizer = TfidfStemVectorizer()

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
            text = line[2]
            date = line[3]
            court = line[4]

            position_count = 0
            for word in vectorizer.build_analyzer()(text):
                if word not in positions:
                    positions[word] = {}
                if doc_id not in positions[word]:
                    positions[word][doc_id] = []
                positions[word][doc_id].append(position_count)
                position_count += 1

            docs.append(doc_id)
            corpus.append(text)
        dataset.close()

    print("building vector space matrix...")
    matrix = (vectorizer.fit_transform(corpus))
    dictionary = vectorizer.vocabulary_

    print("saving to disk...")
    with gzip.open(out_postings, 'wb') as post_file:
        joblib.dump(matrix, post_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open(out_dict, 'wb') as dict_file:
        joblib.dump(vectorizer, dict_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open('docs.txt', 'wb') as doc_file:
        joblib.dump(docs, doc_file, protocol=pickle.HIGHEST_PROTOCOL)
    with gzip.open('positions.txt', 'wb') as posn_file:
        joblib.dump(positions, posn_file, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")
    #print(vectorizer.get_feature_names())

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
