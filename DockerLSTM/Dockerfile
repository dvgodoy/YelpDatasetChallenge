FROM continuumio/miniconda

RUN pip install tensorflow
RUN pip install keras==1.2.2
RUN pip install h5py
RUN pip install flask
RUN pip install cherrypy

EXPOSE 8002

RUN mkdir /webapp
ADD . /webapp

WORKDIR /webapp

CMD python /webapp/server.py
