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

from VentricularImage import VentricularImage;

from os.path import split, join, splitext, isdir, isfile;
from os import mkdir;

from sklearn.neighbors import NearestNeighbors;
from vtk import vtkPolyDataWriter;
from vtk import vtkFloatArray;
from scipy import finfo;
from scipy import empty;
from scipy import where;

class VentricularInterpolator:
    """ DOCSTRING """

    __MRI                       = None;
    __EAM                       = None;
    __kNN                       = None;

    def __init__(self, MRI, EAM):
        # if type(VentricularImage)
        self.__MRI = MRI;
        self.__EAM = EAM;

    def kNearestNeighbours(self, n_neighbors=3, radius=1.0, algorithm='auto', 
                           leaf_size=30, metric='minkowski', p=5, 
                           metric_params=None, n_jobs=1, **kwargs):

        kNN                     = NearestNeighbors(n_neighbors=3, radius=1.0, 
                                                   algorithm='auto', leaf_size=30, 
                                                   metric='minkowski', p=5, 
                                                   metric_params=None, n_jobs=1)
                                                   # **kwargs);

        kNN.fit(EAM.GetOutput().transpose(), MRI.GetOutput().transpose());
        dist, indices           = kNN.kneighbors(MRI.GetOutput().transpose());

        newPolyData             = MRI.GetPolyData();
        newDataArray            = vtkFloatArray();

        EAMScalars              = EAM.GetScalarData();
        BIPS                    = empty((MRI.GetNumberOfPoints()));
        BIPS.fill(0);

        newDataArray.SetNumberOfTuples(MRI.GetNumberOfPoints());
        newDataArray.SetName(EAM.GetPolyData().GetPointData().GetArrayName(0));

        zeroIndex               = where(dist == 0.);

        # If a zero value is found, replace it with the lowest possible number
        if zeroIndex[0].size > 0:
            for j in xrange(zeroIndex[0].size):
                dist[zeroIndex[0][j], zeroIndex[1][j]] = finfo(dist.dtype).eps;

        for i in xrange(MRI.GetNumberOfPoints()):
            dt                  = (1/dist[i,0]) + (1/dist[i,1]) + (1/dist[i,2]);
            BIPS[i]             = ((1/dist[i,:])*EAMScalars[0, indices[i,:]]).sum()/dt;

            newDataArray.SetValue(i, BIPS[i]);


        directory, filename     = split(MRI.GetPath());
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



        newPolyData.GetPointData().AddArray(newDataArray);

        writer.SetInputData(newPolyData);
        writer.Write();

        system("perl -pi -e 's/,/./g' %s " % path);





