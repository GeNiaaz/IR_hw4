.PHONY: search
search: dictionary.txt docs.txt positions.txt postings.txt
	python3 search.py -d dictionary.txt -p postings.txt -q queries/q1.txt -o results.txt
