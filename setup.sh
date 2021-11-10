#! /usr/bin/bash
echo "install poetry if not in path"
if ! command -v poetry &> /dev/null
then
    echo "poetry is missing. Installing ..."
    curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    source "$HOME/.poetry/env"
fi

echo "setting up poetry config"
poetry config virtualenvs.path .venv
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

echo "installing dependencies"
poetry install --no-interaction --no-root

echo "installing libraries"
poetry install --no-interaction
