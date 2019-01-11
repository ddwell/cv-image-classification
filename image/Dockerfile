FROM python:3
ENV PYTHONUNBUFFERED 1
CMD mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
COPY code /code
EXPOSE 8000
CMD python3 manage.py makemigrations
CMD python3 manage.py migrate --run-syncdb
CMD python3 manage.py runserver 0.0.0.0:8000