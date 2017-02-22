"""
    Copyright (C) 2017 - Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""



import os;
import numpy;
import vtk;

import ConfigParser;


# Config = ConfigParser.ConfigParser();

# Config.read("c:\\tomorrow.ini");




# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:





    # try:
    #     val = int(userInput)
    # except ValueError:
    #     print("That's not an int!"

    # try:
    #     val = int(userInput)
    # except ValueError:
    #     print("That's not an int!"

    # try:
    #     val = int(userInput)
    # except ValueError:
    #     print("That's not an int!"

    # try:
    #     val = int(userInput)
    # except ValueError:
    #     print("That's not an int!"

    # try:
    #     val = int(userInput)
    # except ValueError:
    #     print("That's not an int!"





def docstring_example():
    """ 
        Multi-line Docstrings

        Multi-line docstrings consist of a summary line just like a one-line
        docstring, followed by a blank line, followed by a more elaborate
        description. The summary line may be used by automatic indexing tools;
        it is important that it fits on one line and is separated from the rest
        of the docstring by a blank line. The summary line may be on the same
        line as the opening quotes or on the next line. The entire docstring is
        indented the same as the quotes at its first line (see example below).

        Example:
        def complex(real=0.0, imag=0.0):
            '''
                Form a complex number.

                Keyword arguments:
                real -- the real part (default 0.0)
                imag -- the imaginary part (default 0.0)
                ...
            '''


    """



# path = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

def reader_vtk(path):
    reader = vtk.vtkPolyDataReader();
    reader.SetFileName(path);
    reader.Update();

    return reader.GetOutput();



def extract_polygons(polyData):
    rows                = None;
    cols                = None;
    polygons            = None;

    idlist              = vtk.vtkIdList();

    try:
        polys = polyData.GetPolys();
    except RuntimeError:
        print("Tried to call function 'extract_polygons' with a                \
               variable with no 'GetPolys()' method. Check input.");

    for i in range(0, polyData.GetPolys().GetNumberOfCells()):
        success         = polys.GetNextCell(idlist);
        polygon_tuple   = polyData.GetPoint(idlist.GetId(0));

        if polygons is None:
            rows        = len(polygon_tuple);
            cols        = polyData.GetPolys().GetNumberOfCells();
            polygons    = numpy.zeros((rows,cols));
        
        polygons[0,i]   = polygon_tuple[0];
        polygons[1,i]   = polygon_tuple[1];
        polygons[2,i]   = polygon_tuple[2];

    return polygons;



def extract_points(polyData):
    rows                        = None;
    cols                        = None;
    points                      = None;

    try:
        pointVector             = polyData.GetPoints();
    except RuntimeError:
        print("Tried to call function 'extract_points' with a                \
               variable with no 'GetPoints()' method. Check input.");

    if pointVector:
        for i in range(0, pointVector.GetNumberOfPoints()):
            point_tuple         = pointVector.GetPoint(i);

            if points is None:
                rows            = len(point_tuple);
                cols            = pointVector.GetNumberOfPoints();
                points          = numpy.zeros((rows,cols));
            
            points[0,i]         = point_tuple[0];
            points[1,i]         = point_tuple[1];
            points[2,i]         = point_tuple[2];

    return points;



def extract_normals(polyData):
    rows                        = None;
    cols                        = None;
    normals                     = None;

    try:
        normalVector            = polyData.GetPointData().GetNormals();
    except RuntimeError:
        print("Tried to call function 'extract_normals' with a                \
               variable with no 'GetNormals()' method. Check input.");

    if normalVector:
        for i in range(0, normalVector.GetNumberOfTuples()):
            normal_tuple        = normalVector.GetTuple(i);

            if normals is None:
                rows            = len(normal_tuple);
                cols            = normalVector.GetNumberOfTuples();
                normals         = numpy.zeros((rows,cols));
            
            normals[0,i]        = normal_tuple[0];
            normals[1,i]        = normal_tuple[1];
            normals[2,i]        = normal_tuple[2];

    return normals;
















