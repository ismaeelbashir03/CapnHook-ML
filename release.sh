#!/usr/bin/env bash
set -euo pipefail

git checkout main

python -m pip install --upgrade pip
python -m pip install bump-my-version build twine

old=$(cat version.md)
echo "current version $old"

bump-my-version bump \
  --current-version "$old" \
  patch version.md \
  --commit --no-tag \
  --commit-message "bumping version ayo [skip ci]"

new=$(cat version.md)
echo "New version: $new"

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git push origin main

git tag -a "v$new" -m "release v$new"
git push origin "v$new"

# 3. Build and upload to PyPI
python -m build
twine upload dist/*
