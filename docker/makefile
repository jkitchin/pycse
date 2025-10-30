# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  build-3.12     Build Docker image for Python 3.12"
	@echo "  push-3.12      Push Python 3.12 image to Docker Hub"
	@echo "  build          Build Docker image using original Dockerfile (legacy)"
	@echo "  push           Push legacy image to Docker Hub"
	@echo "  run            Run pycse.sh script"
	@echo ""
	@echo "Example: make build-3.12"

# Python 3.12 targets
build-3.12: Dockerfile-3.12
	docker build -f Dockerfile-3.12 -t pycse:3.12 -t pycse:latest .

push-3.12:
	docker tag pycse:3.12 jkitchin/pycse:3.12
	docker tag pycse:3.12 jkitchin/pycse:latest
	docker push jkitchin/pycse:3.12
	docker push jkitchin/pycse:latest

# Legacy targets (original Dockerfile)
build: Dockerfile
	docker build . -t pycse

push:
	docker tag pycse jkitchin/pycse:latest
	docker push jkitchin/pycse:latest

run:
	./pycse.sh

.PHONY: help build-3.12 push-3.12 build push run

