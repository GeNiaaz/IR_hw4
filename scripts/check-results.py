#!/usr/bin/env python3
import itertools
import collections
# Checks results.txt for duplicates
# REFERENCE: https://stackoverflow.com/questions/1541797/how-do-i-check-if-there-are-duplicates-in-a-flat-list
def check_results(file_path):
    with open(file_path, 'r') as file:
        postings = file.read().split()
        if has_duplicates(postings):
            print("DUPLICATES:")
            print(get_duplicates(postings))
        else:
            print("NO DUPLICATES")

def has_duplicates(postings):
    return len(postings) != len(set(postings))

def get_duplicates(postings):
    return [item for item, count in collections.Counter(postings).items() if count > 1]

check_results('results.txt')
