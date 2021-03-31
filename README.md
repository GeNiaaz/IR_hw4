# Requirements

**Input:** legal corpus, set of queries

**Output:** IDs of all relevant documents, sorted by relevance.

External libraries have to be packaged (libraries not installed).

# Setup

## Dataset

1. Install `dataset.zip` from `LumiNUS > CS3245 > Files > Homeworks > Homework #4`.
1. Change dir to project root
1. `unzip dataset.zip`

## Python dependencies

Reference `shell.nix`. Python packages are listed in `my-python-packages`.

# CLI Interface

`indexing`
``` sh
python index.py -i dataset-file -d dictionary-file -p postings-file
```

`searching`
``` sh
python search.py -d dictionary-file -p postings-file -q query-file -o output-file-of-results 
```

- Reuse `hw2`, `hw3` cli parsers. 

- `dataset-file` is a `csv` file. It contains documents to be indexed. You can use `less dataset.csv` to examine contents.

- `query-file` contains a single query.

# Query format

| Query       | Format                            | Example         | Remarks           |
|-------------|-----------------------------------|-----------------|-------------------|
| phrasal     | Enclosed with double-quotes: `""` | "hello world"   | 2-3 words         |
| conjunctive | AND                               | hello AND world | (No `OR` / `NOT`) |
| single-word | self-explanatory                  | hello           |                   |

# Approach

???

# Misc

- Can use `APIs`

- Temp files should be named as `temp-*.*`

- Answer query in `<1 min`.

# Reference

[CS3245 Homework 4](https://www.comp.nus.edu.sg/~cs3245/hw4-intelllex.html)
