FROM tensorflow/tensorflow:2.0.0-py3

LABEL maintainer="gaborpelesz@gmail.com"

ENV FLASK_ENV=production

WORKDIR /

COPY ./app /app

RUN pip3 install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r /app/requirements.txt

CMD [ "python3", "/app/app.py" ]