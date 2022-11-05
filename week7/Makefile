build:
	docker build --network=host -t fs:latest .

run:
	docker run -it --rm --network=host -v $PWD:/app -p 8080:8080 fs:latest /bin/bash

#run_dev: build
#	docker run -it --rm --network=host -v ${PWD}:/main ml_in_prod:week3_latest /bin/bash
#
#test:
#	pytest -ra -s ./tests/
#
#test_data:
#	pytest -ra -s ./tests/test_data.py
#
#test_code:
#	pytest -ra -s ./tests/test_code.py
#
#test_model:
#	pytest -ra -s ./tests/test_model.py
#
#test_all:
#	pytest --cov=image_classification -ra -s tests/
