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

if __name__ == '__main__':
    unittest.main()
