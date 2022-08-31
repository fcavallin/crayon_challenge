FROM tiangolo/uvicorn-gunicorn-fastapi

RUN python -m pip install --upgrade pip

WORKDIR workspace

EXPOSE 8000:80

COPY server.py .
COPY server_dataclasses.py .
COPY transformer_model.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]