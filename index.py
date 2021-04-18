import sys
import getopt
import csv
import pickle
import nltk
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = Stemmer.Stemmer("english")
class TfidfStemVectorizer(TfidfVectorizer):
    def __init__(self):
        super().__init__(self, ngram_range=(1,3))
        self.analyze = super().build_analyzer()
    def build_analyzer(self):
        return lambda doc: (stemmer.stemWord(word) for word in self.analyze(doc))

def usage():
    print("usage: " + sys.argv[0] + " -i dataset-file -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    warning_check = input("are you sure you want to overwrite your index? y/n ")
    if warning_check != "y":
        print("closing...")
        sys.exit(2)

    corpus = []
    docs = []
    vectorizer = TfidfStemVectorizer()

    print("processing dataset...")
    with open('dataset.csv') as dataset:
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

            vectorizer.fit([text])
            docs.append(doc_id)
            corpus.append(text)
        dataset.close()

    print("building vector space model...")
    matrix = vectorizer.transform(corpus)
    dictionary = vectorizer.vocabulary_

    print("saving to disk...")
    with open('postings.txt', 'wb') as postings_file:
        pickle.dump(matrix, postings_file)
    with open('dictionary.txt', 'wb') as dict_file:
        pickle.dump(vectorizer, dict_file)
    with open('docs.txt', 'wb') as doc_file:
        pickle.dump(docs, doc_file)

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
