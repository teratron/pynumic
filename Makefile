

VENV_BIN = .venv/$(shell ls -A .venv | grep -E "bin|Scripts")

list:
	@echo $(VENV_BIN)

install:
	python -m venv .venv
	$(VENV_BIN)/pip install -U pip setuptools
	$(VENV_BIN)/pip install poetry

upgrade: ## upgrade pip
	python -m pip install --upgrade pip

lint: ## lint
	mypy src --ignore-missing-imports
	flake8 src --ignore=$(shell cat .flakeignore)
	black src
	pylint src

clean: ## clean
	poetry cache clear pypi --all
	@rm -rf .pytest_cache/ .mypy_cache/ junit/ build/ dist/
	@find . -not -path "./.venv*" -path "*/__pycache__*" -delete
	@find . -not -path "./.venv*" -path "*/*.egg-info*" -delete

poetry.update: ## update poetry
	poetry update
	poetry self update

poetry.check: ## check poetry
	poetry check

poetry.build: ## build project
	poetry build

poetry.publish: poetry.build ## publish project
	poetry publish

# Publish docs to github pages.
GH_BRANCH   ?= gh-pages
GH_REMOTE   ?= origin
MAIN_BRANCH ?= master
DOCS_DIR     = docs
SOURCE_DIR   = $(DOCS_DIR)/source
BUILD_DIR    = $(DOCS_DIR)/build

html: ## build html docs
	make -C $(DOCS_DIR) html

gh-deploy: html ## deploy docs to github pages
ifeq ($(shell git ls-remote --heads . $(GH_BRANCH) | wc -l), 1)
	@echo "--- Local branch $(GH_BRANCH) exist."
	@echo
	@git branch -D $(GH_BRANCH)
	@echo "--- Deleted branch $(GH_BRANCH)."
endif
	@echo "--- Local branch $(GH_BRANCH) does not exist."
	@echo
	@git checkout --orphan $(GH_BRANCH)
	@echo "--- Created orphan branch $(GH_BRANCH)."
	@git rm -rf $(shell ls -A | grep -vE "Makefile|docs|.git\b|.idea|.fleet|.vscode")
	@echo "--- Removed contents of branch $(GH_BRANCH)."
	@mv -f $(BUILD_DIR)/html/{.[!.],}* $(DOCS_DIR)/.gitignore $(DOCS_DIR)/README.md .
	@echo "--- Moved contents from docs to root of branch $(GH_BRANCH)."
	@rm -rf docs
	@echo "--- Removed docs from branch $(GH_BRANCH)."
	@git add .
	@git commit --allow-empty -m "$(GH_BRANCH)"
	@git push -f $(GH_REMOTE) $(GH_BRANCH)
	@git switch $(MAIN_BRANCH)
	@echo "--- Finished script to create and push $(GH_REMOTE) $(GH_BRANCH)."

set-url: ## git remote set-url origin git@github.com:login/repo.git
	git remote set-url origin git@github.com:zigenzoog/pynumic.git

.PHONY: help install clean build update publish gh-deploy html
help:
	@awk '                                             \
		BEGIN {FS = ":.*?## "}                         \
		/^[a-zA-Z_-]+:.*?## /                          \
		{printf "\033[36m%-24s\033[0m %s\n", $$1, $$2} \
	'                                                  \
	$(MAKEFILE_LIST)

.DEFAULT_GOAL := help
