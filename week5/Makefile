build_base:
	docker build --network=host -f Dockerfile -t base:latest --target base .

run_base: build_base
	docker run -it --rm --network=host base:latest /bin/bash

build_app_streamlit:
	docker build --network=host -f Dockerfile -t app-streamlit:latest --target app-streamlit .

run_app_streamlit: build_app_streamlit
	docker run -it --rm --network=host -e WANDB_API_KEY=${WANDB_API_KEY} app-streamlit:latest