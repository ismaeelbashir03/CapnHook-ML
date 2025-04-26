#!/usr/bin/env bash
set -euo pipefail

git checkout main

python -m pip install --upgrade pip
python -m pip install bump-my-version build twine

old=$(cat version.md)
echo "current version $old"

# 2) Split into major.minor.patch and bump the patch
IFS='.' read -r major minor patch <<< "$old"
patch=$(( patch + 1 ))
new="$major.$minor.$patch"
echo "$new" > version.md
echo "New version: $new"

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git push origin main

git tag -a "v$new" -m "release v$new"
git push origin "v$new"

# 3. Build and upload to PyPI
python -m build
twine upload dist/*
