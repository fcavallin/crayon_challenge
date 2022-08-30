ENV_FILE ?= environment.env

setup_requirements:
	pip install -r requirements.txt

build:
	docker build --build-arg USER_ID=${shell id -u} --build-arg USER_NAME=user -t base_project .

clean_up:
	docker rmi -f base_project

run:
	mkdir -p folder/
	docker run --env-file ${ENV_FILE} --mount type=bind,source=${shell pwd}/folder,target=/home/user/folder -ti base_project

download_resources:
	wget "https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp/download?datasetVersionNumber=1" -P resources

clone_models:
	git lfs install
	git clone https://huggingface.co/prajjwal1/bert-tiny models/huggingface/bert-tiny
	git lfs pull models/huggingface/bert-tiny