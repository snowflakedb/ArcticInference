set -e

if [ ! -f ~/.pypirc ]; then
  echo 'create .pypirc in order to upload to PyPI'
  exit 1
fi

python -m pip install --upgrade build twine

python -m build --wheel

python -m twine upload dist/*
