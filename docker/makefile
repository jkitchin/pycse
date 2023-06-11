build: Dockerfile
	docker build . -t pycse

push:
	docker tag pycse jkitchin/pycse:latest
	docker push jkitchin/pycse:latest

run:
	./pycse.sh

