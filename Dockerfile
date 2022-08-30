FROM python:3.8-slim-buster

ARG USER_ID
ARG USER_NAME=user

ARG USER_HOME=/home/${USER_NAME}

RUN adduser --uid ${USER_ID} --home ${USER_HOME} --shell /bin/bash --disabled-password ${USER_NAME}

USER ${USER_NAME}

WORKDIR ${USER_HOME}

COPY --chown=${USER_ID}:${USER_ID} . ${USER_HOME}

RUN pip3 install -r requirements.txt

CMD ["python3", "-u", "main.py"]