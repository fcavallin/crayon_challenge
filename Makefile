ENV_FILE ?= environment.env

setup_requirements:
	pip install -r requirements.txt

build_training_image:
	docker build --build-arg USER_ID=${shell id -u} --build-arg USER_NAME=user -f docker/training.Dockerfile -t training_image .

clean_up:
	docker rmi -f training_image

start_training:
	mkdir -p models/
	mkdir -p resources/
	docker run --env-file ${ENV_FILE} --mount type=bind,source=${shell pwd}/models,target=/home/user/models --mount type=bind,source=${shell pwd}/resources,target=/home/user/resources -ti training_image

download_resources:
	wget "https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp/download?datasetVersionNumber=1" -P resources

clone_models:
	git lfs install
	git clone https://huggingface.co/prajjwal1/bert-tiny models/bert-tiny
	git lfs pull models/bert-tiny

build_server_image:
	docker build --build-arg USER_ID=${shell id -u} --build-arg USER_NAME=user -f docker/inference.Dockerfile -t server_image .


start_server:
	docker run --env-file ${ENV_FILE} -v ${shell pwd}/models/bert-tiny-finetuned:/app/workspace/models/bert-tiny-finetuned -p 8000:80 server_image
