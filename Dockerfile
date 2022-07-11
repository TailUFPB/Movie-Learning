FROM python:3.10.5

WORKDIR /app
COPY requirements.txt .

RUN pip3 install -r requirements.txt
COPY . /app

EXPOSE 8080

#CMD python3 app.py
CMD gunicorn -b 0.0.0.0:8080 main:app --timeout 6000  --capture-output --log-level debug