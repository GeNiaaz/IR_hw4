import os
import sys
import getopt
import csv
from whoosh import index
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, DATETIME


def usage():
    print("usage: python3 " + sys.argv[0] + " -i dataset-file")


def build_index(dataset_file, output_file_dictionary, output_file_postings):
    print("Indexing dataset...")

    schema = Schema(document_id=ID(stored=True), title=KEYWORD,
                    content=TEXT, date_posted=TEXT, court=KEYWORD)

    if not os.path.exists("index-dir"):
        os.mkdir("index-dir")

    ix = index.create_in("index-dir", schema)
    writer = ix.writer()

    csv.field_size_limit(sys.maxsize)
    with open(dataset_file) as dataset:
        dataset_reader = csv.reader(dataset, delimiter=',')
        counter = 0
        # total = sum(1 for row in dataset_reader)
        total = 17154
        for row in dataset_reader:
            counter += 1
            print(counter, "/", total)
            writer.add_document(document_id=row[0], title=row[1], content=row[2], date_posted=row[3], court=row[4])

    writer.commit()

    print("Done.")

dataset = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        dataset = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if dataset == None:
#if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(dataset, output_file_dictionary, output_file_postings)
