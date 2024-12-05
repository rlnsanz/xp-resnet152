.PHONY: install clean

VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

install: $(VENV_NAME)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV_NAME)/bin/activate:
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip

clean:
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete