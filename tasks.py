#!/usr/bin/env python

"""
This script can be used to build locally or methods from the file can be used in the github actions.
The github actions shall just call methods from this file so we are not locked in the the build server solution.
"""
import contextlib
import os
import shutil

from invoke import task


@task
def check_format_with_black(c, fix=False):
    format_cmd = "black ."
    check_cmd = format_cmd + " --check"
    if fix:
        c.run(format_cmd)
    c.run(check_cmd)


@task
def sort_imports_with_isort(c, fix=False):
    if fix:
        c.run("isort .")
    c.run("isort --check .")


@task
def lint_with_pylint(c):
    c.run("pylint ./src")


@task
def lint_with_flake8(c):
    c.run("flake8 .")


@task
def lint_with_bandit(c):
    c.run("bandit -r src/ --exclude .venv")


@task
def lint_with_mypy(c):
    c.run("mypy")


@task
def clean(c):
    folders = [".venv", ".pytest_cache", ".mypy_cache", "dist", "reports", "coverage"]
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
    files = [".coverage"]
    for file in files:
        with contextlib.suppress(FileNotFoundError):  # -> like ignore_errors=True
            os.remove(file)
