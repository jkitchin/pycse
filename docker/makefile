# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  build-3.12     Build Docker image for Python 3.12"
	@echo "  build-3.13     Build Docker image for Python 3.13 (latest)"
	@echo "  push-3.12      Push Python 3.12 image to Docker Hub"
	@echo "  push-3.13      Push Python 3.13 image to Docker Hub (latest)"
	@echo "  build-all      Build all Python version images"
	@echo "  push-all       Push all Python version images"
	@echo "  run-3.12       Run pycse with Python 3.12"
	@echo "  run-3.13       Run pycse with Python 3.13"
	@echo "  build          Build Docker image using original Dockerfile (legacy)"
	@echo "  push           Push legacy image to Docker Hub"
	@echo "  run            Run pycse.sh script (uses latest)"
	@echo ""
	@echo "Example: make build-3.12"

# Python 3.12 targets
build-3.12: Dockerfile-3.12
	docker build -f Dockerfile-3.12 -t pycse:3.12 .

push-3.12:
	docker tag pycse:3.12 jkitchin/pycse:3.12
	docker push jkitchin/pycse:3.12

# Python 3.13 targets (latest)
build-3.13: Dockerfile-3.13
	docker build -f Dockerfile-3.13 -t pycse:3.13 -t pycse:latest .

push-3.13:
	docker tag pycse:3.13 jkitchin/pycse:3.13
	docker tag pycse:3.13 jkitchin/pycse:latest
	docker push jkitchin/pycse:3.13
	docker push jkitchin/pycse:latest

# Build and push all versions
build-all: build-3.12 build-3.13

push-all: push-3.12 push-3.13

# Legacy targets (original Dockerfile)
build: Dockerfile
	docker build . -t pycse

push:
	docker tag pycse jkitchin/pycse:latest
	docker push jkitchin/pycse:latest

run:
	./pycse.sh

# Run specific Python versions
run-3.12:
	./pycse.sh 3.12

run-3.13:
	./pycse.sh 3.13

.PHONY: help build-3.12 build-3.13 push-3.12 push-3.13 build-all push-all run-3.12 run-3.13 build push run

