# -*- coding: utf-8 -*-

"""
    Copyright (C) 2017 - Universitat Pompeu Fabra
    Author       - Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>
    Contributors - Constantine Butakoff

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

import time

from os import system
from os import mkdir

from os.path import isfile
from os.path import isdir
from os.path import split
from os.path import splitext
from os.path import join

from numpy import int
from numpy import dtype

from scipy.interpolate import Rbf

from numpy.matlib import repmat

from scipy import zeros
from scipy import array
from scipy import asarray
from scipy import mean
from scipy import sqrt
from scipy import pi
from scipy import sin
from scipy import cos
from scipy import tan
from scipy import arccos
from scipy import cumsum
from scipy import where
from scipy import flipud
from scipy import roll
from scipy import dot
from scipy import cross
from scipy import finfo

from scipy.linalg import norm

from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse import find

from vtk import vtkPolyDataReader
from vtk import vtkPolyData
from vtk import vtkPolyDataWriter
from vtk import vtkPoints
from vtk import vtkIdList

from scipy.sparse.linalg import spsolve
from mvpoly.rbf import RBFThinPlateSpline

import utils

class PyQCM(object):
    """
    PyQCM - Extracts a quasi-conformal mapping (QCM) representation of a 3D mesh 
    of an endocardial surface.

    The QCM is performed so that the 3D mesh is fitted into a 2D disk of unit 
    radius. The middle of the basal septum is constrained at the coordinates
    (-1,0), and the apex at the (0,0). The rest of the boundary points are
    linearly spaced to fit the external ring of the disk.
    """

    __path                          = None

    __polydata                      = None
    __points                        = None
    __polygons                      = None

    __batch_size                    = None
    __septum                        = None
    __apex                          = None
    __laplacian                     = None
    __adjacency                     = None
    __boundary                      = None
    __output_polydata               = None
    __homeomorphism                 = None
    __output_path                   = None


    def __init__(self, path, septum=None, apex=None, batch_size=1000, output_path=None):
        """
        Constructor of the object.

        Args:
            path (str): absolute path to the VTK file to be analyzed.
            septum (int): vertex identificator of the middle of the basal septum
            apex (int): vertex identificator of the apical point
            batch_size (int): number of points analyzed at a time
            output_path (str): absoukte path for the resulting VTK file
        """

        self.__path                 = path
        self.__batch_size           = batch_size
        self.__polydata             = utils.polydataReader(self.path)
        self.__points               = utils.vtkPointsToNumpy(self.polydata)
        self.__polygons             = utils.vtkCellsToNumpy(self.polydata)
        self.__laplacian            = utils.cotangentWeightsLaplacianMatrix(self.polydata, self.points, self.polygons)
        self.__adjacency_matrix     = utils.adjacencyMatrix(self.polydata, self.polygons)
        self.__boundary             = utils.boundaryExtractor(self.polydata, self.polygons, self.adjacency_matrix)
        self.__output_path          = utils.outputLocation(self.path, self.output_path)

        landmarks                   = []
        reverse                     = False

        if septum is None:
            if apex is None:
                pass
            else:
                landmarks.append(apex)
                reverse             = True
        else:
            if apex is None:
                landmarks.append(septum)
            else:
                landmarks.append(septum)
                landmarks.append(apex)

        landmarks           = utils.landmarkSelector(self.polydata, 2, landmarks)

        if reverse is True:
            landmarks.reverse()

        boundaryNumber, boundaryId  = utils.closestBoundaryId(self.polydata, landmarks[0], boundary=self.boundary)
        self.__septum               = self.boundary[boundaryNumber][boundaryId]
        self.__apex                 = landmarks[1]

        # A partir de aqui, especificar:
        # * 1 unica boundary valida
        if len(self.boundary) == 1:
            self.__boundary         = self.boundary[0]
            self.__boundary         = roll(self.boundary, -boundaryId)
        else:
            raise Exception("This mapping accepts meshes with only one boundary")

        O           = mean(self.points[:, self.boundary], axis=1)
        OA          = asarray(self.points[:, self.boundary[0]] - O)
        OB          = asarray(self.points[:, self.boundary[1]] - O)
        OC          = asarray(self.points[:, self.apex] - O)
        normal      = cross(OA, OB)

        if dot(OC, normal) < 0:
            self.__boundary = flipud(self.boundary)
            self.__boundary = roll(self.boundary, 1)

        self.__calc_homeomorphism()
        self.__calc_thin_plate_splines()
        self.__write_output()
        utils.vtkWriterSpanishLocale(self.output_path)


    @property
    def path(self):
        """Absolute path to input
        Type: str"""
        return self.__path

    @property
    def output_path(self):
        """Absolute path to output
        Type: str"""
        return self.__output_path

    @property
    def batch_size(self):
        """Number of points analyzed at a time
        Type: int"""
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Number of points analyzed at a time
        Type: int"""
        try:
            self.__batch_size = int(batch_size)
        except:
            raise Exception("Non-number provided")

    @property
    def polydata(self):
        """Number of points analyzed at a time
        Type: vtk.vtkPolyData"""
        return self.__polydata

    @property
    def points(self):
        """Points extracted from the vtk file
        Type: numpy.ndarray"""
        return self.__points

    @property
    def polygons(self):
        """Triplets of point identificators that define each triangle of the mesh
        Type: numpy.ndarray"""
        return self.__polygons

    @property
    def adjacency_matrix(self):
        """Adjacency matrix of the mesh
        Type: scipy.sparse.csr.csr_matrix"""
        return self.__adjacency_matrix

    @output_path.setter
    def output_path(self, output_path):
        """Absolute path to output
        Type: str"""

        if output_path is self.path:
            print(" *  Warning! Overwriting the input file is not permitted.\n"
                  "    Aborting...\n")
            return
        else:
            if self.output_path == self.path:
                print(" *  Warning! The file written to the default location will *not*\n"
                      "    be deleted\n")
            else:
                print(" *  Warning! The file written to the previous working location will \n"
                      "    *not* be deleted\n")

        self.__output_path      = output_path
        self.__write_output()


    @property
    def septum(self):
        """Point identificator of the middle point of the basal septum in the vtk mesh 
        Type: int"""
        return self.__septum

    @property
    def apex(self):
        """Point identificator of the apex in the vtk mesh 
        Type: int"""
        return self.__apex

    @property
    def laplacian(self):
        """Laplacian matrix of the mesh
        Type: scipy.sparse.csr.csr_matrix"""
        return self.__laplacian

    @property
    def boundary(self):
        """Numpy array containing a list of the vertex identifiers of the boundary points in the vtk mesh
        Type: numpy.ndarray"""
        return self.__boundary

    @property
    def homeomorphism(self):
        """Numpy array containing a list of the vertex identifiers of the boundary points in the vtk mesh
        Type: numpy.ndarray"""
        return self.__homeomorphism

    @property
    def output_polydata(self):
        """Polydata containing the QCM of the calculated transformation
        Type: vtk.vtkPolyData"""
        return self.__output_polydata


    def __calc_homeomorphism(self):
        """Calculation of the conformal mapping between LV endocardial surfaces 
        and a 2D disk. The approximation of the Laplacian computation is based 
        on cotangent weights.
        """

        start = time.time()

        if (self.laplacian is not None) and (self.boundary is not None):
            numPoints               = self.polydata.GetNumberOfPoints()
            diagonal                = self.laplacian.sum(0)
            diagonalSparse          = spdiags(diagonal, 0, numPoints, numPoints)
            homeomorphism_laplacian = diagonalSparse - self.laplacian

            # Finds non-zero elements in the laplacian matrix
            (nzj, nzi)              = self.laplacian.nonzero()

            for point in self.boundary:
                positions           = where(nzi==point)[0]

                homeomorphism_laplacian[nzi[positions], nzj[positions]] = 0
                homeomorphism_laplacian[point, point]                   = 1

            # Finds a distribution of the boundary points around a circle
            boundaryNext            = roll(self.boundary, -1)
            boundaryNextPoints      = self.points[:, boundaryNext]
            distanceToNext          = boundaryNextPoints - self.points[:, self.boundary]

            euclideanNorm           = sqrt((distanceToNext**2).sum(0))
            perimeter               = euclideanNorm.sum()

            fraction                = euclideanNorm/perimeter

            angles                  = cumsum(2*pi*fraction)
            angles                  = roll(angles, 1)
            angles[0]               = 0

            # Creates the constrain for the homeomorphism
            Z                       = zeros((2, angles.size))
            Z[0,:]                  = cos(angles)
            Z[1,:]                  = sin(angles)

            boundaryConstrain       = zeros((2, self.polydata.GetNumberOfPoints()))
            boundaryConstrain[:, self.boundary] = Z

            self.__homeomorphism    = spsolve(homeomorphism_laplacian, 
                                              boundaryConstrain.transpose()).transpose()


    def __calc_thin_plate_splines(self, batch_size=None):
        """Correction of anatomical landmarks based on Thin-Plate Splines (TPS) 
        that relaxes the established conformal mapping to quasi-conformal.
        """

        if (self.homeomorphism is None) or (self.apex is None):
            raise Exception("The homeomorphic transformation could not be calculated")

        if batch_size is None:
            batch_size  = self.batch_size

        boundaryPoints  = self.homeomorphism[:,self.boundary]
        source          = zeros((boundaryPoints.shape[0],
                                 boundaryPoints.shape[1] + 1))
        destination     = zeros((boundaryPoints.shape[0],
                                 boundaryPoints.shape[1] + 1))

        source[:, 0:source.shape[1] - 1]        = boundaryPoints
        source[:, source.shape[1] - 1]          = self.homeomorphism[:, self.apex]

        destination[:, 0:source.shape[1] - 1]   = boundaryPoints
        destination[:, 0:source.shape[1] - 1]   = boundaryPoints

        # For a faster calculation, the mvpoly package has been used. The
        # Thin Plate Splines has been calculated using the X coordinate of
        # the points in the real part of a complex number and the Y
        # coordinate of the point as the imaginary part. After the 
        # interpolation, the separate real and imaginary parts have been
        # recovered, encoding the new (X,Y) positions after the relaxation.
        x = source[0,:]
        y = source[1,:]
        d = destination[0,:] + 1j*destination[1,:]

        thinPlateInterpolation = RBFThinPlateSpline(x,y,d)
        result                      = zeros((1, self.homeomorphism.shape[1]), dtype='complex128')

        # To avoid memory error, smaller batch sizes are used. The result is
        # independent of the batch size.
        for i in range(0, self.homeomorphism.shape[1], batch_size):
            if (i + batch_size) >= self.homeomorphism.shape[1]:
                result[:,i:]        = thinPlateInterpolation(self.homeomorphism[0,i:],
                                                             self.homeomorphism[1,i:])
            else:
                result[:,i:(i+batch_size)]  = thinPlateInterpolation(self.homeomorphism[0,i:(i+batch_size)],
                                                                     self.homeomorphism[1,i:(i+batch_size)])

        # Recover the new (X,Y) coordinates from the real and imaginary parts
        # of the results.
        self.__homeomorphism[0,:] = result.real
        self.__homeomorphism[1,:] = result.imag


    def flip_boundary(self):
        """The method for extracting the boundaries can result in clockwise or 
        counterclockwise boundaries. This method is used to flip the rotation 
        direction if needed.
        """
        self.__boundary         = flipud(self.boundary)
        self.__boundary         = roll(self.boundary, 1)


    def __write_output(self):
        """Writes the vtkPolyData onject to file. The scalars from the original
        vtk input file are reused.
        """
        if ((self.homeomorphism is None) or (self.points is None) or (self.polygons is None)):
            raise Exception("Something went wrong. Check the input")

        newPolyData         = vtkPolyData()
        newPointData        = vtkPoints()
        writer              = vtkPolyDataWriter()
        writer.SetFileName(self.output_path)

        for i in xrange(self.polydata.GetNumberOfPoints()):
            newPointData.InsertPoint(i, (self.homeomorphism[0, i], self.homeomorphism[1, i], 0.0))

        newPolyData.SetPoints(newPointData)
        newPolyData.SetPolys(self.polydata.GetPolys())

        if self.polydata.GetPointData().GetScalars() is None:
            newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetArray(0))
        else:
            newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetScalars())
            
        bool_scalars = False

        for i in range(self.polydata.GetPointData().GetNumberOfArrays()):
            if self.polydata.GetPointData().GetScalars() is None:
                if self.polydata.GetPointData().GetArray(i).GetName() == 'Bipolar':
                    newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetArray(i))
                    bool_scalars = True
                else:
                    newPolyData.GetPointData().AddArray(self.polydata.GetPointData().GetArray(i))
            else:
                if (self.polydata.GetPointData().GetArray(i).GetName() 
                    == self.polydata.GetPointData().GetScalars().GetName()):
                    newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetArray(i))
                else:
                    newPolyData.GetPointData().AddArray(self.polydata.GetPointData().GetArray(i))
            
            
        writer.SetInputData(newPolyData)
        writer.Write()

        self.__output_polydata  = newPolyData


    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path + "'.\n"
        s = s + "Number of dimensions: " + str(self.polydata.GetPoints().GetData().GetNumberOfComponents()) + "\n"
        s = s + "Number of points: " + str(self.polydata.GetNumberOfPoints()) + "\n"
        s = s + "Number of polygons: " + str(self.polydata.GetNumberOfCells()) + "\n"
        s = s + "Output file location: " + str(self.output_path)
        return s