FROM ubuntu:16.04
FROM python:3.5-slim

# Install all the required packages for the model
RUN pip install numpy
RUN pip install keras
RUN pip install pillow
RUN pip install tensorflow

# Arguments definition (Passed in the command line) for github login - Required because it is a private repository
ARG username
ARG password

#Install Github related packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

#Current working directory
WORKDIR /home/ec2-user

# Cloning github repository onto AWS instance
RUN git clone https://${username}:${password}@github.com/khyatisinha/ap_pretrained.git

#Current working directory
WORKDIR /home/ec2-user/ap_pretrained

EXPOSE 80

# Command to run the model
CMD ["python", "./keras_resnet.py"]
