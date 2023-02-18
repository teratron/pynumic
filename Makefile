# Main scripts for the project.

VENV     = .venv
VENV_BIN = $(VENV)/$(shell ls -A $(VENV) | grep -E "bin|Scripts")

setup: ## setup virtual environment and poetry
	@python -m venv $(VENV)
	@make install

install: ## install poetry
ifeq ($(shell ls -A | grep -E $(VENV)), $(VENV))
	@$(VENV_BIN)/python -m pip install --upgrade pip
	@$(VENV_BIN)/pip install -U pip setuptools
	@$(VENV_BIN)/pip install poetry
	@poetry check
	@poetry lock
	@poetry install
	@$(VENV_BIN)/pip install -e . # dev
else
	@make install
endif

update: ## update poetry
	@poetry update
	@poetry self update

check: ## check poetry
	@poetry check

build: check clean ## build project
	@poetry build

publish: build ## publish project
	@poetry publish

lint: ## lint project
	@mypy src --ignore-missing-imports
	@flake8 src --ignore=$(shell cat .flakeignore)
	@black src
	@pylint src

#test-find:
#	@echo $(shell ls -AR | grep -E "*__pycache__*|*.mypy_cache*|*.egg-info*")
#	@echo $(shell ls -R $(shell ls -A | grep -vE "$(VENV)|.git*|.idea|.mypy_cache|.pytest_cache|.egg-info|__pycache__|build|dist|junit") \
#	| grep -E ".mypy_cache|.pytest_cache|.egg-info|__pycache__|build|dist|junit")

clean: ## clean
	@poetry cache clear pypi --all
	@rm -rf .pytest_cache/ .mypy_cache/ junit/ build/ dist/
ifeq ($(OS), Linux)
	@find . -not -path "./$(VENV)" -path "*/__pycache__*" -delete
	@find . -not -path "./$(VENV)" -path "*/*.egg-info*" -delete
endif

# Publish docs to github pages.

MAIN_BRANCH ?= master
GH_BRANCH   ?= gh-pages
GH_REMOTE   ?= origin
DOCS_DIR     = docs
BUILD_DIR    = $(DOCS_DIR)/build

gh-deploy: ## deploy docs to github pages
	@make -C $(DOCS_DIR) html
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
	@rm -rf $(shell ls -A | grep -vE "Makefile|.git\b|$(DOCS_DIR)|$(VENV)|.idea|.fleet|.vscode")
	@echo "--- Removed contents of branch $(GH_BRANCH)."
	@mv -f $(BUILD_DIR)/html/{.[!.],}* $(DOCS_DIR)/.gitignore $(DOCS_DIR)/README.md .
	@echo "--- Moved contents from docs to root of branch $(GH_BRANCH)."
	@rm -rf $(DOCS_DIR)
	@echo "--- Removed docs from branch $(GH_BRANCH)."
	@git add .
	@git reset -- .gitignore Makefile
	@git commit --allow-empty -m "$(GH_BRANCH)"
	@git push -f $(GH_REMOTE) $(GH_BRANCH)
	@git switch $(MAIN_BRANCH)
	@echo "--- Finished script to create and push $(GH_REMOTE) $(GH_BRANCH)."

set-url: ## git remote set-url origin git@github.com:login/repo.git
	@git remote set-url origin git@github.com:zigenzoog/pynumic.git

git-hooks:
	@mkdir .githooks || true
	@git config --global core.hooksPath .githooks

env-prepare: # создать .env-файл для секретов
	@cp -n .env.sample .env || true

.PHONY: help setup install update check build publish lint clean gh-deploy set-url
help:
	@awk '                                             \
		BEGIN {FS = ":.*?## "}                         \
		/^[a-zA-Z_-]+:.*?## /                          \
		{printf "\033[36m%-24s\033[0m %s\n", $$1, $$2} \
	'                                                  \
	$(MAKEFILE_LIST)

.DEFAULT_GOAL := help
