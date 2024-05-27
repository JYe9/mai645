# Use the latest Ubuntu base image
FROM ubuntu:latest

# Set metadata for the image
LABEL authors="jye"

# Update package lists and install necessary packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install Python libraries
RUN pip3 install transforms3d torch torchvision opencv-python

# Set the entrypoint command
ENTRYPOINT ["top", "-b"]
