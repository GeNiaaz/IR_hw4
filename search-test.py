#!/usr/bin/env python3

from search import *
import unittest

class TestParseQuery(unittest.TestCase):

    def test_single_word(self):
        self.assertEqual(parse_query("abc"), ["abc"])

    def test_conjunction_2_words(self):
        self.assertEqual(parse_query("mouse AND cat"), ["mouse", "cat"])

    def test_bag_of_words(self):
        self.assertEqual(parse_query("a bag of words"), ["a bag of words"])

    def test_phrasal_query(self):
        self.assertEqual(parse_query('"phrasal query"'), ['"phrasal query"'])

    def test_phrasal_query_and_bog(self):
        self.assertEqual(parse_query('"phrasal query" AND a bag o\' words'), ['"phrasal query"', "a bag o' words"])

    def test_rocchio_2_docs(self):
        alpha = 1
        beta = 1
        query_vec = [1, 2, 3, 4]
        doc_vec_1 = [2, 2, 2, 2]
        doc_vec_2 = [4, 4, 4, 4]

        self.assertEqual(rocchio_calculation(alpha, beta, query_vec, doc_vec_1, doc_vec_2), [4, 5, 6, 7])

    def test_rocchio_5_docs(self):
        alpha = 1
        beta = 1
        query_vec = [1, 2, 3, 4]
        doc_vec_1 = [2, 2, 2, 2]
        doc_vec_2 = [3, 3, 3, 3]
        doc_vec_3 = [4, 4, 4, 4]
        doc_vec_4 = [5, 5, 5, 5]
        doc_vec_5 = [6, 6, 6, 6]

        self.assertEqual(rocchio_calculation(alpha, beta, query_vec, doc_vec_1, doc_vec_2,
                                             doc_vec_3, doc_vec_4, doc_vec_5), [5, 6, 7, 8])

    def test_rocchio_alpha_beta(self):
        alpha = 2
        beta = 5
        query_vec = [1, 2, 3, 4]
        doc_vec_1 = [2, 2, 2, 2]
        doc_vec_2 = [4, 4, 4, 4]

        self.assertEqual(rocchio_calculation(alpha, beta, query_vec, doc_vec_1, doc_vec_2), [17, 19, 21, 23])


    """ Testing the AND thing """

    def test_AND(self):
        ls1 = [1,2,3,4,5,6,7,8,10]
        ls2 = [2,4,6,7,9,10]

        self.assertEqual(process_AND(ls1, ls2), [2,4,6,7,10])

if __name__ == '__main__':
    unittest.main()
