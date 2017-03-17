# Copyright (C) 2017 - Universitat Pompeu Fabra
# Author - Guillermo Jimenez-Perez  <guillermo.jim.per@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


FROM ubuntu:16.10

MAINTAINER Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>

################################## APT-GET #####################################
RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends       \
                            libvtk6.3=6.3.0+dfsg1-1build1                      \
                            python=2.7.11-2                                    \
                            python-numpy=1:1.11.1~rc1-1ubuntu1                 \
                            python-scipy=0.17.1-1                              \
                            python-pip=8.1.2-2ubuntu0.1                        \
                            git=1:2.9.3-1                                      \
                            && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mvpoly==0.97.5

# WORKDIR /work


RUN groupadd -r host && useradd -r -g host host && usermod -u 1000 host
USER host





################################ MINICONDA #####################################
# RUN wget -q https://repo.continuum.io/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh \
#        -O /home/miniconda.sh                                                 

# RUN chmod +x /home/miniconda.sh                                                              


# RUN apt-get update && apt-get install -y                                       \
#                           wget=1.18-2ubuntu1                                   \
#                           bzip2=1.0.6-8build1                                  \
#                           && rm -rf /var/lib/apt/lists/*

# RUN   /home/miniconda.sh -b -p /conda                                          \
#   && rm /home/miniconda.sh                                                     \
#   && export PATH="/conda/bin:$PATH"                                            \
#   && /conda/bin/conda install -c clinicalgraphics vtk=7.1.0 -y --quiet         \
#   && /conda/bin/conda install jupyter=1.0.0                                    \
#                               numpy=1.11.3                                     \
#                               scipy=0.18.1                                     \
#                               pytorch=0.1.9                                    \
#                               torchvision=0.1.7                                \
#                               cuda80=1.0                                       \
#                               -c soumith                                       \
#                               -y --quiet

# RUN   /home/miniconda.sh -b -p /conda                                          \
#   && rm /home/miniconda.sh                                                     \
#   && export PATH="/conda/bin:$PATH"                                            \
#   && /conda/bin/conda install -c clinicalgraphics vtk=7.1.0 -y --quiet         \
#   && /conda/bin/conda install jupyter=1.0.0                                    \
#                               numpy=1.11.3                                     \
#                               scipy=0.18.1                                     \
#                               # vtk=7.1.0                                      \
#                               -y --quiet                                       \
#   && /conda/bin/conda install pytorch=0.1.9                                    \
#                               torchvision=0.1.7                                \
#                               cuda80=1.0                                       \
#                               -c soumith -y --quiet
################################ MINICONDA #####################################




