FROM gurobi/python:latest

WORKDIR /app_swc

ADD requirements.txt /app_swc/requirements.txt
RUN pip install -r requirements.txt

COPY . /app_swc

CMD ["python", "-u","app.py"]