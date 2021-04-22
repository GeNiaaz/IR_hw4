This is the README file for A0181059A-A0205280R-A0200007H-A0200161E's submission

== Python Version ==

We're using Python Version 3.9.0 for this assignment.

== General Notes about this assignment ==

This program implements a document search engine for legal case retrieval purposes, by constructing a vector space model and a positional index to represent a set of legal documents as a matrix of weighted vectors. The program implements searching using either free text, phrasal queries (handled using the positional index), or a combination of the two, using Boolean operators (specifically, the AND operator).

The program makes extensive use of the scikit-learn machine learning library, with the NumPy and SciPy scientific computing libraries to handle complex operations involving matrices in the vector space.

<index.py>

This file handles the indexing of dataset.csv, which contains the entire set of legal documents for this program. To index the contents of the dataset file, the entire file is iterated over, to retrieve the title, court and text content of each case in the file. These three fields are then tokenized, concatenated, case-folded, and stemmed using a Snowball stemmer, and finally, stored according to their case number, to be processed and fitted to a matrix representing a vector space model later.

First, we iterate through the text content of each case, to store the positions of each word present in each case. The positions of each word in each case is stored in a dictionary of dictionaries; each unique word acts as the key to a nested dictionary, in which all the documents in which that word appears are the keys to a list containing the positions of that word in that document.

Afterwards, the scikit-learn function TfidfVectorizer is used to learn the vocabulary used in the raw text collected earlier, and subsequently convert that raw text into a mxn vector space matrix of tf-idf features, with m rows to represent each of the m documents in dataset.csv, and n columns for each unique word detected in the dataset. The function automatically calculates and normalizes the tf-idf scores for each word in each document. We used a ltc.ltc weighting scheme for the tf-idf scores, making the assumption that the dataset provided is static and not subject to change. The function also smoothens the idf weights by adding a value of 1 to all existing document frequencies.

The output of this process leaves us with four important functions/structures; a mxn vector space matrix, a list representing the mapping of document IDs to row numbers in the mxn matrix, a dictionary of dictionaries representing a positional index, and a vectorizer (the TfidfVectorizer function used earlier). The same vectorizer must be used in searching to convert our queries into 1xn query vectors; thus, we save these four structures by pickling them, thus completing the indexing process.

<search.py>

The searching process begins with the unpickling of the four structures (the vector space matrix, the list containing the document ID - matrix row number mapping, the positional index and the vectorizer). The query file is then processed to retrieve both the query string itself, and a list of document IDs as relevance judgments.

Firstly, the query string is processed to separate AND operators, phrasal queries (which are found between single or double apostrophes) and free text within the query. Once phrases in the query are identified, the documents that match each phrase within the query are retrieved using the positional index created earlier, using the following procedure:

1. For each query word in the phrase, find the set of relevant docs that contain the word and merge (find intersection of) these sets.
2. For each doc ID in the merged set, check if the phrase exist using the relative positional indexes of the query words.
3. Return the list of docs containing the query phrase.

If multiple phrases exist in the query, separated by an AND operator, the list of documents returned by each phrase are intersected with each other to only return the common documents between those phrases. Thus, the phrasal components of the query are handled first. If the query also contains non-phrasal elements (i.e. free text), they are separately handled using the vector-space matrix created earlier.

For free text, the vectorizer created during indexing is used to transform the free-text query into a 1xn vector (where n represents the length of vocabulary present in the dataset), with normalized tf-idf scores for each word present in the query. Words that exist in the dataset vocabulary, but not in the query, will naturally have a td-idf score of 0. The scikit-learn function cosine_similarity is then used to compute the normalized cosine score for that vector, and subsequently find the document vectors in the vector-space matrix that have the closest score (i.e. the greatest cosine similarity) with the score of the query vector. The document vectors in the matrix are then sorted in order of their cosine similarity with the query vector, and written to the output file in that order. Document vectors with a similarity score of 0 are not written to output.

We experimented with query refinement by implementing the Rocchio algorithm using the relevance judgments. Given the list of documents that are known to be relevant in the query, we retrieved the 1xn document vectors for those documents from our vector space matrix, and calculated the mean of those vectors to retrieve their centroid. We then assigned a weight of 0.7 to the query vector and a weight of 0.3 to the centroid, and added them together to obtain a theoretically optimized query vector. We chose to assign more weight to the query vector, as we knew the number of relevance judgments would be low; we did not want to risk skewing the query vector significantly to potentially obtain an incorrect query vector. 


== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

README.txt: High-level documentation about our solution 
index.py: Source code for indexing 
search.py: Source code for searching
dictionary.txt: Indexed dictionary of terms in the legal case corpus
postings.txt: Indexed postings with tf-idf score and positional indexes of terms in the legal case corpus

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] We, A0181059A-A0205280R-A0200007H-A0200161E, certify that we have followed the CS3245 Information Retrieval class guidelines for homework assignments.  In particular, we expressly vow that we have followed the Facebook rule in discussing with others in doing the assignment and did not take notes (digital or printed) from the discussions.  


== References ==

CS3245 lecture slides :)
scikit-learn.org for scikit-learn documentation
StackOverflow for pesky debugging issues
