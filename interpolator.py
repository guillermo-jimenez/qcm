"""
    Copyright (C) 2017 - Universitat Pompeu Fabra
    Author - Guillermo Jimenez-Perez  <guillermo.jim.per@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

# from VentricularImage import VentricularQCM;

from os.path import split, join, splitext, isdir, isfile;
from os import mkdir, system;

from sklearn.neighbors import NearestNeighbors;

from vtk import vtkPolyDataWriter;
from vtk import vtkFloatArray;

from scipy import finfo;
from scipy import zeros;
# from scipy import empty;
from scipy import where;

class VentricularInterpolator(object):
    """ DOCSTRING """

    __image_high_res            = None;
    __image_low_res             = None;

    __MRI                       = None;
    __EAM                       = None;
    __kNN                       = None;
    __interpolated_polydata     = None;

    @property
    def image_high_res(self):
        return self.__image_high_res;

    @image_high_res.setter
    def image_high_res(self, image_high_res):
        self.__image_high_res   = image_high_res;

    @property
    def image_low_res(self):
        return self.__image_low_res;

    @image_low_res.setter
    def image_low_res(self, image_low_res):
        self.__image_low_res    = image_low_res;

    @property
    def kNN(self):
        return self.__kNN;

    def __init__(self, image_high_res, image_low_res):
        self.__image_high_res   = image_high_res;
        self.__image_low_res    = image_low_res;


    def kNearestNeighbours(self, n_neighbors=3, radius=1.0, algorithm='auto', 
                           leaf_size=30, metric='minkowski', p=5, 
                           metric_params=None, n_jobs=1, **kwargs):

        kNN                     = NearestNeighbors(n_neighbors=3, radius=1.0, 
                                                   algorithm='auto', leaf_size=30, 
                                                   metric='minkowski', p=5, 
                                                   metric_params=None, n_jobs=1)
                                                   # **kwargs);

        kNN.fit(self.image_low_res.QCM_points.transpose(), 
                self.image_high_res.QCM_points.transpose());

        dist, indices           = kNN.kneighbors(self.image_high_res.QCM_points.transpose());

        newPolyData             = self.image_high_res.polydata;
        newDataArrays           = [];

        for i in xrange(len(self.image_low_res.scalars_names)):
            newDataArrays.append(vtkFloatArray());
            newDataArrays[i].SetNumberOfTuples(self.image_high_res.npoints);
            newDataArrays[i].SetName(self.image_low_res.scalars_names[i]);

        lr_scalars              = self.image_low_res.scalars;
        BIPS                    = zeros((len(self.image_low_res.scalars_names), 
                                         self.image_high_res.npoints));
        # BIPS                    = empty((self.image_high_res.npoints));
        # BIPS.fill(0);

        # newDataArray.SetNumberOfTuples(self.image_high_res.npoints);
        # newDataArray.SetName(self.image_low_res.polydata.GetPointData().GetArrayName(0));

        zeroIndex               = where(dist == 0.);

        # If a zero value is found, replace it with the lowest possible number
        if zeroIndex[0].size > 0:
            for j in xrange(zeroIndex[0].size):
                dist[zeroIndex[0][j], zeroIndex[1][j]] = finfo(dist.dtype).eps;

        for j in xrange(len(self.image_low_res.scalars_names)):
            for i in xrange(self.image_high_res.npoints):
                dt                  = (1/dist[i,0]) + (1/dist[i,1]) + (1/dist[i,2]);
                BIPS[j, i]          = ((1/dist[i,:])*lr_scalars[self.image_low_res.scalars_names[j]][j, indices[i,:]]).sum()/dt;

                newDataArrays[j].SetValue(i, BIPS[j, i]);


        directory, filename     = split(self.image_high_res.path);
        filename, extension     = splitext(filename);

        writer                  = vtkPolyDataWriter();

        if isdir(join(directory, 'Fusion')):
            path                = join(directory, 'Fusion', str(filename + '_Fusion' + extension));
            writer.SetFileName(path);
        else:
            mkdir(join(directory, 'Fusion'));

            if isdir(join(directory, 'Fusion')):
                path            = join(directory, 'Fusion', str(filename + '_Fusion' + extension));
                writer.SetFileName(path);
            else:
                path            = join(directory, str(filename + '_Fusion' + extension));
                writer.SetFileName(path);


        for dataArray in newDataArrays:
            newPolyData.GetPointData().AddArray(dataArray);

        writer.SetInputData(newPolyData);
        writer.Write();

        system("perl -pi -e 's/,/./g' %s " % path);





