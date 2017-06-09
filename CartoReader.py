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


class CartoReader(object):
    """Explicación de la clase
    """

    __path          = None;
    __binfile       = None;
    __output        = None;

    def __init__(self, path):
        """BaseImage(path)

        Analyzes a ventricular image in vtk format, extracting the point, mesh
        and scalar information from it. Requires a path as a string to be
        initialized.
        """

        if isfile(path):
            self.__path         = path;
        else:
            raise RuntimeError("File does not exist");

        self.__read_to_vtk();

    @property
    def path(self):
        return self.__path;

    # @path.setter
    # def path(self, path):
    #     if isfile(path):
    #         if self.path is not None:
    #             print("Overwritting existing data on variable...")
    #         else:
    #             print("Loading data...")

    #         self.__init__(path);
    #     else:
    #         raise RuntimeError("File does not exist");

    # @property
    # def output(self):
    #     """Testing docstring of attribute"""
    #     return self.__polydata;

    def __read_to_vtk(self):
        attributesDict                      = dict();
        tags                                = dict();
        tags['[GeneralAttributes]']         = False;
        tags['[VerticesSection]']           = False;
        tags['[TrianglesSection]']          = False;
        tags['[VerticesColorsSection]']     = False;
        tags['[VerticesAttributesSection]'] = False;
        
        if (splitext(self.path)[1] == '.mesh'):
            if isfile(self.path):
                data = open(self.path, 'r').read().splitlines();
                data[:] = (line for line in data if line != '');
            else:
                print("The file does not exist.");
                return
        else:
            print("Only '.mesh' files are accepted. Set a new file path.");
            return

        for i in xrange(len(data)):
            if ((data[i][0] == '#') or (';')):
                continue
            elif data[i][0] == '[':
                for i in tags:
                    tags[i]                 = False;

                if data[i] in tags:
                    tags[data[i]]           = True;
                    continue
                else:
                    raise RuntimeError('Tag not considered. Contact maintainer.\nLine states: ' + data[i])

            if tags['[GeneralAttributes]']:
                splits = data[i].split();
                attributesDict[splits[0]] = [splits[i] for i in range(2, len(splits))];

            elif tags['[VerticesSection]']:

            elif tags['[TrianglesSection]']:

            elif tags['[VerticesColorsSection]']:

            elif tags['[VerticesAttributesSection]']:



writer = vtk.vtkPolyDataWriter()
polydata = vtk.vtkPolyData()
points = vtk.vtkPoints()
pointNormals = vtk.vtkDoubleArray()
pointNormals.SetNumberOfComponents(3)
pointNormals.SetNumberOfTuples(points.GetNumberOfPoints())



    def GetOutput(self):
        return self.__output;

    def __read_polydata(self):
        reader                  = vtkPolyDataReader();
        reader.SetFileName(self.path);
        reader.Update();

        polydata                = reader.GetOutput();
        polydata.BuildLinks();

        self.__npoints          = polydata.GetNumberOfPoints();
        self.__npolygons        = polydata.GetNumberOfPolys();
        self.__ndim             = polydata.GetPoints().GetData().GetNumberOfComponents();
        self.__polydata         = polydata;


    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path;
        return s;

