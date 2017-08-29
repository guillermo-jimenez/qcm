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

# from abc import ABCMeta, abstractmethod;

from os.path import isfile;

from numpy import int;
from numpy import dtype;

from scipy import zeros;
from scipy import array;

from vtk import vtkPolyDataReader;


class BaseImage(object):
    """
    Data extraction from VTK file, regardless of the specific image 
    (ventricle, atrium; endocardium, epicardium)



    Attributes
    ----------

    path:               str
        path to the image file

    ndim:               int
        dimensionality of the VTK image

    npoints:            int
        number of points in the VTK file

    npolygons:          int
        number of polygons in the VTK file

    nedges_mesh:        int
        number of edges of the mesh in the input VTK file. Determines the
        shape of the mesh; whether it is triangular, quadrilateral or of a
        higher order

    nscalars:           int
        number of scalar vectors in the input VTK file

    polydata:           vtkPolyData
        link to the original vtkPolyData stored in the 'path' variable

    points:             numpy.ndarray
        coordinates of the point in a numpy array, in shape (ndim, npoints)

    polygons:           numpy.ndarray
        point identifiers of each of the triangles of the mesh, 
        in shape (nedges_mesh, npolygons). If a point of a mesh cell wants to
        be accessed, the specific point ID has to be passed to the points array:

        >>> # (being ob the BaseImage object)
        >>> ob.nedges_mesh # The mesh is triangular
        3
        >>> ob.polygons[:,35] # Let's check the point IDs of the 36th triangle
        array([286, 715, 69])
        >>> # If we wanted to check the spatial coordinates of the first node
        >>> # of the 36th triangle, that is, the 287th node of the set, it
        >>> # could be accessed by passing the 1st node of the 36th triangle
        >>> # to the points array:
        >>> point_of_interest = ob.points[:, ob.polygons[0, 35]]
        array([  56.53070068, -101.97899628,  179.80900574]) 

    scalars:            numpy.ndarray
        structured numpy array containing the information of the scalars
        associated to each point coordinate, in shape (nscalars, npoints). It
        allows for specific attribute calls. E.g:  if a VTK file with two scalar
        fields is analyzed, the first being called 'scalars' and the second
        'LAT', the following statement is valid:

        >>> # (being ob the BaseImage object)
        >>> ob.scalars['LAT']
        array([ 0.52275603,  0.94302633,  0.81044762, ...,  0.02766091,
                0.18389358,  0.62247573])

    scalars_names:      tuple
        wrapper for 'scalars.dtype.names'. Provides the names of the scalar 
        arrays contained in the input VTK file

    normals:            numpy.ndarray
        numpy array containing the normals of each point, in shape 
        (ndim, npoints)



    Returns
    -------

    output:         BaseImage
        BaseImage object containing the extracted information of the input VTK
        file



    See Also
    --------

    VentricularEndocardium



    Example
    -------

    >>> import BaseImage
    >>> import os
    >>> path = os.path("/path/to/image.vtk")
    >>> image1 = BaseImage.BaseImage(path)
    >>> image1.scalars
    array([624, 263, 142 ... 124, 412, 416])

    """

    # __metaclass__               = ABCMeta;

    __path                      = None;
    __polydata                  = None;
    __points                    = None;
    __polygons                  = None;
    __scalars                   = None;
    __normals                   = None;
    __npoints                   = None;
    __npolygons                 = None;
    __ndim                      = None;
    __nscalars                  = None;
    __nedges_mesh               = None;
    __scalars_names             = None;

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

        self.__read_polydata();
        self.__read_points();
        self.__read_polygons();
        self.__read_normals();
        self.__read_scalars();
        # self.__calc_laplacian_matrix();

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

    @property
    def polydata(self):
        """Testing docstring of attribute"""
        return self.__polydata;

    @property
    def ndim(self):
        return self.__ndim;

    @property
    def nedges_mesh(self):
        return self.__nedges_mesh;

    @property
    def points(self):
        return self.__points;

    @property
    def npoints(self):
        return self.__npoints;

    @property
    def polygons(self):
        return self.__polygons;

    @property
    def npolygons(self):
        return self.__npolygons;

    @property
    def scalars(self):
        return self.__scalars;

    @property
    def scalars_names(self):
        return self.__scalars_names;

    @property
    def normals(self):
        return self.__normals;


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

    def __read_polygons(self):
        rows                    = None;
        cols                    = None;
        polygons                = None;

        polys                   = self.polydata.GetPolys();

        for i in xrange(self.polydata.GetNumberOfCells()):
            triangle            = self.polydata.GetCell(i);
            pointIds            = triangle.GetPointIds();

            if polygons is None:
                rows            = pointIds.GetNumberOfIds();
                cols            = self.polydata.GetNumberOfCells();
                polygons        = zeros((rows,cols), dtype=int);
            
            polygons[0,i]       = pointIds.GetId(0);
            polygons[1,i]       = pointIds.GetId(1);
            polygons[2,i]       = pointIds.GetId(2);

        self.__polygons         = polygons;

    def __read_points(self):
        rows                    = None;
        cols                    = None;
        points                  = None;

        pointVector             = self.polydata.GetPoints();

        if pointVector:
            for i in range(0, pointVector.GetNumberOfPoints()):
                point_tuple     = pointVector.GetPoint(i);

                if points is None:
                    rows        = len(point_tuple);
                    cols        = pointVector.GetNumberOfPoints();
                    points      = zeros((rows,cols));
                
                points[0,i]     = point_tuple[0];
                points[1,i]     = point_tuple[1];
                points[2,i]     = point_tuple[2];

        self.__points           = points;

    def __read_normals(self):
        rows                    = None;
        cols                    = None;
        normals                 = None;

        normalVector            = self.polydata.GetPointData().GetNormals();

        if normalVector:
            for i in range(0, normalVector.GetNumberOfTuples()):
                normalTuple     = normalVector.GetTuple(i);

                if normals is None:
                    rows        = len(normalTuple);
                    cols        = normalVector.GetNumberOfTuples();
                    normals     = zeros((rows,cols));
                
                normals[0,i]    = normalTuple[0];
                normals[1,i]    = normalTuple[1];
                normals[2,i]    = normalTuple[2];

        self.__normals          = normals;

    def __read_scalars(self):
        point_data              = self.polydata.GetPointData();
        scalars                 = None;
        nscalars                = 0;

        for i in xrange(point_data.GetNumberOfArrays()):
            scalar_name         = point_data.GetArrayName(i);
            scalar_array        = point_data.GetArray(i);
            scalar_dtype        = scalar_array.GetDataTypeAsString();

            if scalar_array.GetNumberOfComponents() is 1:
                aux_dtype       = dtype([(scalar_name, scalar_dtype)])
                aux             = zeros((1, scalar_array.GetNumberOfTuples()), 
                                        dtype=scalar_dtype);
                nscalars        = nscalars + 1;

                for j in xrange(scalar_array.GetNumberOfTuples()):
                    aux[0, j]   = scalar_array.GetTuple1(j);

                aux             = array(aux, dtype=aux_dtype);

                if scalars is None:
                    scalars     = aux;
                else:
                    scalars     = self.__join_struct_arrays(scalars, aux)

        self.__nscalars         = nscalars;
        self.__scalars          = scalars;
        self.__scalars_names    = scalars.dtype.names;

    def __join_struct_arrays(array1, array2, *args):
        new_dtype = [];

        descriptor = [];
        for field in array1.dtype.names:
            (typ, _) = array1.dtype.fields[field];
            descriptor.append((field, typ));
        new_dtype.extend(tuple(descriptor));

        descriptor = [];
        for field in array2.dtype.names:
            (typ, _) = array2.dtype.fields[field];
            descriptor.append((field, typ));
        new_dtype.extend(tuple(descriptor));
        
        for arr in args:
            descriptor = [];
            for field in arr.dtype.names:
                (typ, _) = arr.dtype.fields[field];
                descriptor.append((field, typ));
            new_dtype.extend(tuple(descriptor));

        new_rec_array = np.zeros(len(arrays[0]), dtype = new_dtype);

        for name in array1.dtype.names:
            new_rec_array[name] = array1[name];

        for name in array2.dtype.names:
            new_rec_array[name] = array2[name];

        for arr in args:
            for name in arr.dtype.names:
                new_rec_array[name] = arr[name];

        return new_rec_array;

    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path + "'.\n";
        s = s + "Number of dimensions: " + str(self.ndim) + "\n";
        s = s + "Number of points: " + str(self.npoints) + "\n";
        s = s + "Number of polygons: " + str(self.npolygons) + "\n";
        s = s + "Number of edges of the polygons: " + str(self.nedges_mesh) + "\n";
        s = s + "Scalar information: " + str(self.scalars_names);
        return s;

