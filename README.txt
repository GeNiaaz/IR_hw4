This is the README file for A0181059A-A0205280R-A0200007H-A0200161E's submission

== Python Version ==

We're using Python Version 3.9.0 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

<index.py>

[Indexing - Abhijit]


<search.py>

[Cosine sim - Abhijit]

[Rocchio - Geniaaz]

[AND - Noel]

Phrasal Queries
1. For each query word in the phrase, find the set of relevant docs that contain the word and merge (find intersection of) these sets in increasing size.
2. For each doc ID in the merged set, for each query word, get the positional indexes list and shift each index back by the query word's index position in the phrase, and merge with the shifted list of every subsequent query word. This will uncover the docs that contain the exact query phrase, with the merged positions being the positional index of the first query word.
3. The list of docs containing the query phrase is returned.


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

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>
