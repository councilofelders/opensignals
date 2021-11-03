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

if [ -n "`$SHELL -c 'echo $ZSH_VERSION'`" ]; then
    shell="zsh"
elif [ -n "`$SHELL -c 'echo $BASH_VERSION'`" ]; then
    shell="bash"
else
   # assume something else
   exit
fi

echo "setup tab completion for invoke for $shell"
# see http://docs.pyinvoke.org/en/stable/invoke.html#shell-tab-completion
poetry run invoke --print-completion-script $shell > .invoke-completion.sh

echo "To activate the tab completion, run the following command:"
echo ""
echo "    $ source .invoke-completion.sh"
echo ""
