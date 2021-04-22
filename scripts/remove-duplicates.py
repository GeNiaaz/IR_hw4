#!/usr/bin/env python3

import collections

def remove_duplicates(results):
    return [item for item, count in collections.Counter(results).items() if count == 1]

# test
print(remove_duplicates(["1", "1", "2", "3"]))
