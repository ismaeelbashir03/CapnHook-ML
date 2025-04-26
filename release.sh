#!/usr/bin/env bash
set -euo pipefail

git checkout main

python -m pip install --upgrade pip
python -m pip install build twine auditwheel scikit-build-core-conan

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

# build and upload to PyPI
# must do it this way as you cannot upload a linux_*.whl wheel to PyPI (https://peps.python.org/pep-0513/#rationale)
python -m build --wheel --no-isolation

# https://stackoverflow.com/questions/59451069/binary-wheel-cant-be-uploaded-on-pypi-using-twine
for whl in dist/*linux_x86_64.whl; do
  auditwheel repair "$whl" --plat manylinux2014_x86_64 -w dist/
done

rm -f dist/*linux_x86_64.whl

twine upload dist/*manylinux2014_x86_64.whl
