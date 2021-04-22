.PHONY: index
index: index.py dataset.csv dictionary.txt docs.txt postings.txt
	python3 index.py -i dataset.csv -d dictionary.txt -p postings.txt


.PHONY: search
search: search.py dictionary.txt docs.txt positions.txt postings.txt
	python3 search.py -d dictionary.txt -p postings.txt -q queries/q1.txt -o results.txt

.PHONY: bundle
bundle:
	python3 scripts/bundle.py

.PHONY: check-duplicates
check-duplicates:
	python3 scripts/check-results.py
