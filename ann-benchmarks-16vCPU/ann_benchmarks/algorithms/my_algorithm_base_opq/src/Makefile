pypi: dist
	twine upload dist/*

dist:
	-rm python_bindings/dist/*
	pip install build
	cd python_bindings && python3 -m build --sdist .

test:
	python3 -m unittest discover --start-directory python_bindings/tests/python --pattern "bindings_test*.py"

clean:
	rm -rf python_bindings/*.egg-info python_bindings/build python_bindings/dist python_bindings/tmp python_bindings/var python_bindings/tests/__pycache__ python_bindings/hnswlib.cpython*.so

.PHONY: dist test clean pypi
