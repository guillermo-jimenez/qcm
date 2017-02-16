FROM ubuntu:16.10

MAINTAINER Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>

RUN apt-get update && apt-get install -y 								\
							libvtk6.3=6.3.0+dfsg1-1build1				\
							python3=3.5.1-4								\
							&& rm -rf /var/lib/apt/lists/*

RUN groupadd -r host && useradd -r -g host host && usermod -u 1000 host
USER host



RUN export PYTHONPATH=/home/doriad/bin/VTK/lib:/home/doriad/bin/VTK/lib/site-packages:/home/doriad/bin/VTK/Wrapping/Python:$PYTHONPATH