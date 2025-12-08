# You can also run multiple in sequence, e.g. `make clean lint test serve-coverage-report`

build:
	bash run.sh build

clean:
	bash run.sh clean

docker-build:
	bash run.sh docker-build

docker-run:
	bash run.sh docker-run
	
help:
	bash run.sh help

install:
	bash run.sh install

lint:
	bash run.sh lint

lint-ci:
	bash run.sh lint:ci

serve-coverage-report:
	bash run.sh serve-coverage-report

test-ci:
	bash run.sh test:ci

test-quick:
	bash run.sh test:quick

test:
	bash run.sh run-tests
