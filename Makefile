SERVICES=python

# Build the python service image and force remove previous instance
build:
	docker-compose build --force-rm --pull --no-cache $(SERVICES)

# build gpu enabled
build-gpu:
	docker build --tag blokkosvision:latest --file Dockerfile.gpu ./

build-base:
	docker build --tag tensorflow-cv:1.0.0-gpu --file ./docker-build-tf-opencv/Dockerfile.cvgpu

# Push to docker hub
push:
	docker-compose push $(SERVICES)

# Run all the service containers with compose (docker-compose up) and force recreate container for python service
run:
	docker-compose up --force-recreate -d $(SERVICES)

# Run on GPU
run-gpu:
	docker run -it --rm --gpus all -p 3000:3000 blokkosvision:latest

# Run developement environment (make run with docker-compose.dev.yml)
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --force-recreate -d $(SERVICES)

# Get into the python service's container shell
exec:
	docker-compose exec python sh
	
# Turn down running docker-compose services
down:
	docker-compose down --volumes
	docker system prune -f
