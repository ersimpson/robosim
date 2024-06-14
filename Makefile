install: requirements.txt
	pip-sync requirements.txt

requirements.txt: requirements.in dev-requirements.in
	pip-compile requirements.in dev-requirements.in --output-file requirements.txt
