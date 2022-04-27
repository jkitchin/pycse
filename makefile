flake8:
	pre-commit run flake8

black:
	pre-commit run black

pylint:
	#pre-commit run pylint
	pylint --rcfile=setup.cfg pycse
