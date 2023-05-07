
FROM python:3.10-slim-bullseye


WORKDIR /app

COPY app.py ./app.py
COPY requirements.txt ./requirements.txt
COPY model.pkl ./model.pkl


RUN pip install -r requirements.txt


EXPOSE 5000



CMD python app.py
