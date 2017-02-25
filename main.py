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



# import os;
# import torch;
# import numpy;
# import scipy;
# import vtk;
# import time;

# import ConfigParser;


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





# def docstring_example():
#     """ 
#         Multi-line Docstrings

#         Multi-line docstrings consist of a summary line just like a one-line
#         docstring, followed by a blank line, followed by a more elaborate
#         description. The summary line may be used by automatic indexing tools;
#         it is important that it fits on one line and is separated from the rest
#         of the docstring by a blank line. The summary line may be on the same
#         line as the opening quotes or on the next line. The entire docstring is
#         indented the same as the quotes at its first line (see example below).

#         Example:
#         def complex(real=0.0, imag=0.0):
#             '''
#                 Form a complex number.

#                 Keyword arguments:
#                 real -- the real part (default 0.0)
#                 imag -- the imaginary part (default 0.0)
#                 ...
#             '''


#     """


# import torch;
from __future__ import division

import os;
import numpy;
import scipy;

from numpy.matlib import repmat;
from scipy.sparse import coo_matrix;
from scipy.sparse import csr_matrix;
from scipy.sparse import csc_matrix;
from scipy.sparse import lil_matrix;
from scipy.sparse import csgraph;

from scipy.special import cotdg;

import time;
import vtk;



def reader_vtk(path):
    reader = vtk.vtkPolyDataReader();
    reader.SetFileName(path);
    reader.Update();

    return reader.GetOutput();



def extract_polygons(polyData):
    rows                = None;
    cols                = None;
    polygons            = None;

    try:
        polys = polyData.GetPolys();
    except RuntimeError:
        print("Tried to call function 'extract_polygons' with a                \
               variable with no 'GetPolys()' method. Check input.");

    for i in range(0, polyData.GetNumberOfCells()):
        triangle            = polyData.GetCell(i);
        pointIds            = triangle.GetPointIds();

        if polygons is None:
            rows            = pointIds.GetNumberOfIds();
            cols            = polyData.GetNumberOfCells();
            polygons        = scipy.zeros((rows,cols), dtype=numpy.int);
        
        # for j in range(0, pointIds.GetNumberOfIds()):
        #     polygons[j,i]   = pointIds.GetId(j);

        polygons[0,i]       = pointIds.GetId(0);
        polygons[1,i]       = pointIds.GetId(1);
        polygons[2,i]       = pointIds.GetId(2);

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
                points          = scipy.zeros((rows,cols));
            
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
                normals         = scipy.zeros((rows,cols));
            
            normals[0,i]        = normal_tuple[0];
            normals[1,i]        = normal_tuple[1];
            normals[2,i]        = normal_tuple[2];

    return normals;


# def laplacian_matrix(polyData):
#     num_dims        = 3;
#     num_points      = polyData.GetNumberOfPoints();
#     num_polygons    = polyData.GetNumberOfCells();

#     sparse_matrix   = scipy.sparse.coo_matrix((num_points, num_points));

#     dist_p2_p1      = scipy.zeros((num_dims, num_polygons));
#     dist_p3_p1      = scipy.zeros((num_dims, num_polygons));

#     dist_p1_p2      = scipy.zeros((num_dims, num_polygons));
#     dist_p3_p2      = scipy.zeros((num_dims, num_polygons));

#     dist_p1_p3      = scipy.zeros((num_dims, num_polygons));
#     dist_p2_p3      = scipy.zeros((num_dims, num_polygons));

#     for i in range(0, num_polygons):
#         triangle    = polyData.GetCell(i);
#         points      = triangle.GetPoints();

#         point_1     = points.GetPoint(0);
#         point_2     = points.GetPoint(1);
#         point_3     = points.GetPoint(2);

#         point_id_1  = triangle.GetPointId(0);
#         point_id_2  = triangle.GetPointId(1);
#         point_id_3  = triangle.GetPointId(2);

#         dist_p2_p1[0, i] = point_2[0] - point_1[0];
#         dist_p2_p1[1, i] = point_2[1] - point_1[1];
#         dist_p2_p1[2, i] = point_2[2] - point_1[2];

#         dist_p3_p1[0, i] = point_3[0] - point_1[0];
#         dist_p3_p1[1, i] = point_3[1] - point_1[1];
#         dist_p3_p1[2, i] = point_3[2] - point_1[2];

#         dist_p1_p2[0, i] = point_1[0] - point_2[0];
#         dist_p1_p2[1, i] = point_1[1] - point_2[1];
#         dist_p1_p2[2, i] = point_1[2] - point_2[2];

#         dist_p3_p2[0, i] = point_3[0] - point_2[0];
#         dist_p3_p2[1, i] = point_3[1] - point_2[1];
#         dist_p3_p2[2, i] = point_3[2] - point_2[2];

#         dist_p1_p3[0, i] = point_1[0] - point_3[0];
#         dist_p1_p3[1, i] = point_1[1] - point_3[1];
#         dist_p1_p3[2, i] = point_1[2] - point_3[2];

#         dist_p1_p3[0, i] = point_1[0] - point_3[0];
#         dist_p1_p3[1, i] = point_1[1] - point_3[1];
#         dist_p1_p3[2, i] = point_1[2] - point_3[2];


def laplacian_matrix(point_data, polygon_data):
    num_dims            = polygon_data.shape[0];
    num_points          = point_data.shape[1];
    num_polygons        = polygon_data.shape[1];

    sparse_matrix       = scipy.sparse.coo_matrix((num_points, num_points));

    for i in range(0, num_dims):
        i1              = (i + 0)%3;
        i2              = (i + 1)%3;
        i3              = (i + 2)%3;

        dist_p2_p1      = point_data[:, polygon_data[i2, :]]        \
                          - point_data[:, polygon_data[i1, :]];
        dist_p3_p1      = point_data[:, polygon_data[i3, :]]        \
                          - point_data[:, polygon_data[i1, :]];

        dist_p2_p1      = dist_p2_p1 / numpy.matlib.repmat(scipy.sqrt((dist_p2_p1**2).sum(0)), 3, 1);
        dist_p3_p1      = dist_p3_p1 / numpy.matlib.repmat(scipy.sqrt((dist_p3_p1**2).sum(0)), 3, 1);

        angles          = scipy.arccos((dist_p2_p1 * dist_p3_p1).sum(0));

        iter_data1      = scipy.sparse.coo_matrix((1/scipy.tan(angles),        \
                                              (polygon_data[i2,:],             \
                                              polygon_data[i3,:])),            \
                                              shape=(num_points, num_points));

        iter_data2      = scipy.sparse.coo_matrix((1/scipy.tan(angles),        \
                                                  (polygon_data[i3,:],         \
                                                  polygon_data[i2,:])),        \
                                                  shape=(num_points, num_points));

        sparse_matrix   = sparse_matrix + iter_data1 + iter_data2;


    # d = sparse_matrix.sum(0).todense();
    diagonal            = sparse_matrix.sum(0);
    diagonal_sparse     = scipy.sparse.spdiags(d,0,num_points,num_points);
    laplacian_matrix    = diagonal_sparse - sparse_matrix;


    return laplacian_matrix;



    # dist_p2_p1[1, i] = point_2[1] - point_1[1];
    # dist_p2_p1[2, i] = point_2[2] - point_1[2];

    # dist_p3_p1  = point_3[0] - point_1[0];
    # dist_p3_p1[1, i] = point_3[1] - point_1[1];
    # dist_p3_p1[2, i] = point_3[2] - point_1[2];






        # Tengo que: para cada 



        # pp[:, ];


# def laplacian_matrix(point_data, polygon_data):
#     sparse_matrix = scipy.sparse.coo_matrix((point_data.size, point_data.size));
#     num_dims      = polygon_data.shape[0];
#     num_polys     = polygon_data.shape[1];

#     pp = scipy.zeros((num_dims, num_polys));
#     qq = scipy.zeros((num_dims, num_polys));

#     for i in range(0, num_dims):
#         i1          = (i + 0)%3;
#         i2          = (i + 1)%3;
#         i3          = (i + 2)%3;

#         for j in range(0, num)
#         dist_p2_p1[:, ]



# for i=1:3
#    i1 = mod(i-1,3)+1;
#    i2 = mod(i  ,3)+1;
#    i3 = mod(i+1,3)+1;
#    pp = vertex(:,faces(i2,:)) - vertex(:,faces(i1,:));
#    qq = vertex(:,faces(i3,:)) - vertex(:,faces(i1,:));
#    % normalize the vectors
#    pp = pp ./ repmat( sqrt(sum(pp.^2,1)), [3 1] );
#    qq = qq ./ repmat( sqrt(sum(qq.^2,1)), [3 1] );
#    % compute angles
#    ang = acos(sum(pp.*qq,1));
#    W = W + make_sparse(faces(i2,:),faces(i3,:), 1 ./ tan(ang), n, n );
#    W = W + make_sparse(faces(i3,:),faces(i2,:), 1 ./ tan(ang), n, n );
# end




    # Intended Usage:
    # * COO is a fast format for constructing sparse matrices
    # * Once a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations
    # * By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together. This facilitates efficient construction of finite element matrices and the like. (see example)



# path      = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

path            = os.path.join("/home/bee/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

polyData        = reader_vtk(path);
point_data      = extract_points(polyData);
polygon_data    = extract_polygons(polyData);








