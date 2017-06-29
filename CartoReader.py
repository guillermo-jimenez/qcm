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


from __future__ import division
from os.path import isfile;
from os.path import splitext;
from os.path import split;
from os.path import isdir;
from os.path import join;
from vtk import vtkPolyData;
from vtk import vtkPoints; 
from vtk import vtkCellArray; 
from vtk import vtkDoubleArray;
from vtk import vtkTriangle;


import vtk
import time
import numpy

class CartoReader(object):
    """Explicación de la clase
    """

    __path                                = None;
#     __binfile                             = None;
#     __output                              = None;

    __attributes                          = dict();
    __tags                                = dict();
    __output_path                         = __path;
    
    __tags['[GeneralAttributes]']         = False;
    __tags['[VerticesSection]']           = False;
    __tags['[TrianglesSection]']          = False;
    __tags['[VerticesColorsSection]']     = False;
    __tags['[VerticesAttributesSection]'] = False;
    
    __polydata = vtkPolyData();

    def __init__(self, path, output_path=None):
        """BaseImage(path)

        Analyzes a ventricular image in vtk format, extracting the point, mesh
        and scalar information from it. Requires a path as a string to be
        initialized.
        """

        if isfile(path):
            self.__path         = path;
        else:
            raise RuntimeError("File does not exist");


        if output_path is not None:
            self.__output_path  = output_path;
        else:
            print(" * Writing to default location: ")

            path                = None;
            directory, filename = split(self.path);
            filename, extension = splitext(filename);
            
            if isdir(join(directory, 'VTK')):
                self.__output_path            = join(directory, 'VTK', str(filename + '.vtk'));
            else:
                mkdir(join(directory, 'VTK'));

                if isdir(join(directory, 'VTK')):
                    self.__output_path        = join(directory, 'VTK', str(filename + '.vtk'));
                else:
                    print("  !-> Could not create output directory. Writing in input directory")
                    self.__output_path        = join(directory, str(filename + '.vtk'));
            
            print("  --> " + self.output_path);

        self.__read_to_vtk();

    @property
    def path(self):
        return self.__path;

    @property
    def attributes(self):
        return self.__attributes;

    @property
    def output(self):
        return self.__polydata;

    @property
    def output_path(self):
        return self.__output_path;
    
    def __read_to_vtk(self):
        writer       = vtkPolyDataWriter();

        points       = vtkPoints();
        cells        = vtkCellArray();

        pointNormals = vtkDoubleArray();
        cellNormals  = vtkDoubleArray();
        cellArray = vtkDoubleArray();

        signals      = [];
        boolSignals  = False;

        pointNormals.SetNumberOfComponents(3);
        pointNormals.SetNumberOfTuples(points.GetNumberOfPoints());
        pointNormals.SetName('PointNormals');

        cellNormals.SetNumberOfComponents(3);
        cellNormals.SetNumberOfTuples(points.GetNumberOfPoints());
        cellNormals.SetName('CellNormals');

        if (splitext(self.path)[1] == '.mesh'):
            if isfile(self.path):
                data = open(self.path, 'r').read().splitlines();
#                 data[:] = (line for line in data if line != '');
            else:
                print("The file does not exist.");
                return;
            
        else:
            print("Only '.mesh' files are accepted. Set a new file path.");
            return;

        for i in xrange(len(data)):
            if data[i] == '':
                continue;
                
            elif ((data[i][0] == '#') or (data[i][0] == ';')):
                continue;

            elif data[i][0] == '[':
                for j in self.__tags:
                    self.__tags[j]          = False;

                if data[i] in self.__tags:
                    self.__tags[data[i]]    = True;
                    continue;

                else:
                    raise RuntimeError('Tag not considered. Contact maintainer.\nLine states: ' + data[i]);

            if self.__tags['[GeneralAttributes]']:
                splits = data[i].split();
                self.__attributes[splits[0]] = [splits[i] for i in range(2, len(splits))];

            elif boolSignals == False:
                boolSignals = True;

                nPoints      = int(self.__attributes['NumVertex'][0]);
                nColors      = int(self.__attributes['NumVertexColors'][0]);
                nAttrib      = int(self.__attributes['NumVertexAttributes'][0]);

                for j in xrange(nColors + nAttrib):
                    cellArray = vtkDoubleArray();
                    cellArray.SetNumberOfComponents(1);
                    if j < nColors:
                        cellArray.SetName(self.__attributes['ColorsNames'][j]);
                    else:
                        cellArray.SetName(self.__attributes['VertexAttributesNames'][j - nColors]);

                    signals.append(cellArray);

            if self.__tags['[VerticesSection]']:
                points.InsertPoint(int(data[i].split()[0]), (float(data[i].split()[2]),
                                                             float(data[i].split()[3]),
                                                             float(data[i].split()[4])));

                pointNormals.InsertNextTuple3(float(data[i].split()[5]),
                                              float(data[i].split()[6]),
                                              float(data[i].split()[7]));

                ### FALTA EL GROUPID DE ESTE VERTICESSECTION

            elif self.__tags['[TrianglesSection]']:
                triangle = vtkTriangle();

                triangle.GetPointIds().SetId(0, int(data[i].split()[2]));
                triangle.GetPointIds().SetId(1, int(data[i].split()[3]));
                triangle.GetPointIds().SetId(2, int(data[i].split()[4]));

                cells.InsertNextCell(triangle);

                cellNormals.InsertNextTuple3(float(data[i].split()[5]),
                                             float(data[i].split()[6]),
                                             float(data[i].split()[7]));

                ### FALTA EL GROUPID DE ESTE TRIANGLESSECTION
                    
            elif self.__tags['[VerticesColorsSection]']:
                for j in range(2, len(data[i].split())):
                    signals[j-2].InsertNextTuple1(float(data[i].split()[j]));

            elif self.__tags['[VerticesAttributesSection]']:
                for j in range(2, len(data[i].split())):
                    signals[nColors + j - 2].InsertNextTuple1(float(data[i].split()[j]));

        writer.SetFileName(self.output_path);

        self.__polydata.SetPoints(points);
        self.__polydata.SetPolys(cells);

        self.__polydata.GetPointData().SetNormals(pointNormals);
        self.__polydata.GetCellData().SetNormals(cellNormals);

        for array in signals:
            if array.GetName() == 'Bipolar':
                self.__polydata.GetPointData().SetScalars(array);
            else:
                self.__polydata.GetPointData().AddArray(array);

        writer.SetInputData(self.__polydata);
        writer.SetFileName(self.output_path)
        writer.Write()

    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path;
        return s;

