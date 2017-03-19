FROM ubuntu:14.04
RUN apt-get -y update
RUN apt-get -y install curl

# PIP
RUN apt-get install -y python python-dev python-distribute python-pip
RUN pip install py4j
RUN pip install flask
RUN pip install cherrypy
RUN pip install numpy

# JAVA
ARG JAVA_ARCHIVE=http://download.oracle.com/otn-pub/java/jdk/8u102-b14/server-jre-8u102-linux-x64.tar.gz
ENV JAVA_HOME /usr/local/jdk1.8.0_102

ENV PATH $PATH:$JAVA_HOME/bin
RUN curl -jksSLH "Cookie: oraclelicense=accept-securebackup-cookie;" ${JAVA_ARCHIVE} | tar -xz -C /usr/local/ && ln -s $JAVA_HOME /usr/local/java

# SPARK
ARG SPARK_ARCHIVE=http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz
ENV SPARK_HOME /usr/local/spark-2.0.0-bin-hadoop2.7

ENV PATH $PATH:${SPARK_HOME}/bin
ENV PYTHONPATH $PYTHONPATH:${SPARK_HOME}/python:${SPARK_HOME}/python/build
RUN curl -s ${SPARK_ARCHIVE} | tar -xz -C /usr/local/

RUN cp ${SPARK_HOME}/conf/log4j.properties.template ${SPARK_HOME}/conf/log4j.properties
RUN echo "log4j.rootCategory=ERROR,console" >> ${SPARK_HOME}/conf/log4j.properties

EXPOSE 80

RUN mkdir /webapp
ADD . /webapp

WORKDIR ${SPARK_HOME}

CMD python /webapp/server.py

# docker run -p 8001:8001 -i -t sparkweb