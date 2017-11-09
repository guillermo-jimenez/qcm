# -*- coding: utf-8 -*-

# Copyright (C) 2017 - Universitat Pompeu Fabra
# Author - Guillermo Jimenez-Perez  <guillermo.jim.per@gmail.com>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

from os import system
from os import mkdir

from os.path import isfile
from os.path import isdir
from os.path import split
from os.path import splitext
from os.path import join

from numpy import int
from numpy import dtype

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

from scipy.linalg import norm

from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.sparse import find

from vtk import vtkPolyDataReader
from vtk import vtkPolyData
from vtk import vtkPolyDataWriter
from vtk import vtkPoints
from vtk import vtkIdList
from vtk import vtkPlane
from vtk import vtkClipPolyData

from scipy.sparse.linalg import spsolve
from mvpoly.rbf import RBFThinPlateSpline

from PointPicker import PointSelector

import utils


class EpiQCM(object):
    __path                          = None

    __polydata                      = None
    __points                        = None
    __polygons                      = None

    __boundary                      = None
    __anterior                      = None
    __posterior                     = None
    __apex                          = None

    __LV_laplacian                  = None
    __LV_boundary                   = None
    __LV_polydata                   = None
    __LV_homeomorphism              = None
    __LV_apex                       = None
    __LV_anterior                   = None
    __LV_posterior                  = None

    __RV_boundary                   = None
    __RV_laplacian                  = None
    __RV_polydata                   = None
    __RV_homeomorphism              = None
    __RV_apex                       = None
    __RV_anterior                   = None
    __RV_posterior                  = None

    __output_path                   = None


    def __init__(self, path, anterior=None, posterior=None, apex=None, batch_size=1000, output_path=None):
        """ """

        print("TO-DO: DOCUMENTATION")

        self.__path                 = path
        self.__batch_size           = batch_size
        self.__polydata             = utils.polydataReader(self.path)
        # self.__points               = utils.vtkPointsToNumpy(self.polydata)
        # self.__polygons             = utils.vtkCellsToNumpy(self.polydata)
        # self.__adjacency_matrix     = utils.adjacencyMatrix(self.polydata, self.polygons)
        # self.__laplacian            = utils.cotangentWeightsLaplacianMatrix(self.polydata, self.points, self.polygons)
        # self.__boundary             = utils.boundaryExtractor(self.polydata, self.polygons, self.adjacency_matrix)
        # self.__output_path          = utils.outputLocation(self.path, self.output_path)

        landmarks                   = []
        # reverse                     = False
        reverse                     = False
        center                      = False

        if anterior is None:
            if apex is None:
                if posterior is None:
                    pass
                else:
                    landmarks.append(posterior)
                    reverse             = True
            else:
                if posterior is None:
                    landmarks.append(apex)
                    center              = True
                else:
                    # Bien
                    landmarks.append(posterior)
                    landmarks.append(apex)
                    reverse             = True
        else:
            if apex is None:
                if posterior is None:
                    pass
                else:
                    landmarks.append(posterior)
                    reverse             = True
            else:
                if posterior is None:
                    # Bien
                    landmarks.append(anterior)
                    landmarks.append(apex)
                else:
                    # Bien
                    landmarks.append(anterior)
                    landmarks.append(apex)
                    landmarks.append(posterior)

        landmarks           = utils.landmarkSelector(self.polydata, 3, landmarks)

        if reverse is True:
            landmarks.reverse()

        # Divide polydata into two:
        




        posteriorNumber, posteriorId    = utils.closestBoundaryId(self.polydata, landmarks[0], boundary=self.boundary)
        anteriorNumber, anteriorId      = utils.closestBoundaryId(self.polydata, landmarks[2], boundary=self.boundary)
        self.__anterior                 = self.boundary[boundaryNumber][boundaryId]
        self.__apex                     = landmarks[1]
        self.__posterior                = self.boundary[boundaryNumber][boundaryId]

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

        self.__anterior             = anterior
        self.__posterior            = posterior
        self.__apex                 = apex

        # Reading the input VTK file
        if self.polydata is None:
            self.__polydata = utils.polydataReader(path)

        # Establishing an output path
        if output_path is None:
            self.__output_path      = self.path
        else:
            if output_path is self.path:
                print(" *  Output path provided coincides with input path.\n"+
                      "    Overwriting the input file is not permitted.\n"+
                      "    Writing in the default location...\n")
                self.__output_path  = self.path
            else:
                self.__output_path  = output_path

        self.__points               = utils.vtkPointsToNumpy(self.polydata)
        self.__polygons             = utils.vtkCellsToNumpy(self.polydata)

        # Calculate landmarks if not provided
        self.__calc_landmarks()
        self.__closest_boundary_point()
        self.__calc_boundary()
        self.__rearrange()
        self.__calc_laplacian()
        self.__calc_homeomorphism()
        self.__calc_thin_plate_splines()
        self.__write_output()

    @property
    def path(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__path

    @property
    def output_path(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__output_path

    @property
    def polydata(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__polydata

    @property
    def points(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__points

    @property
    def polygons(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__polygons

    ##########################################################################
    # ¿¿ELIMINAR??
    ##########################################################################
    # @output_path.setter
    # def output_path(self, output_path):
    #     """ """

    #     print("TO-DO: DOCUMENTATION")

    #     if output_path is self.path:
    #         print(" *  Warning! Overwriting the input file is not permitted.\n"
    #               "    Aborting...\n")
    #         return
    #     else:
    #         if self.output_path == self.path:
    #             print(" *  Warning! The file written to the default location will *not*\n"
    #                   "    be deleted\n")
    #         else:
    #             print(" *  Warning! The file written to the previous working location will \n"
    #                   "    *not* be deleted\n")

    #     self.__output_path      = output_path
    #     self.__write_output()
    ##########################################################################
    # ¿¿ELIMINAR??
    ##########################################################################


    @property
    def anterior(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__anterior

    @property
    def posterior(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__posterior

    @property
    def apex(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__apex

    ##########################################################################
    # ¿¿ELIMINAR??
    ##########################################################################
    # @apex.setter
    # def apex(self, apex):
    #     """ """

    #     print("TO-DO: DOCUMENTATION")

    #     if apex >= self.polydata.GetNumberOfPoints():
    #         raise RuntimeError("Apical point provided is out of bounds")

    #     self.__apex             = apex
    #     self.__calc_homeomorphism()
    #     self.__write_output()
    ##########################################################################
    # ¿¿ELIMINAR??
    ##########################################################################

    @property
    def laplacian(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__laplacian

    @property
    def boundary(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__boundary

    @property
    def boundary_points(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.points[:, self.boundary]

    @property
    def homeomorphism(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__homeomorphism

    @property
    def adjacency_matrix(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__adjacency_matrix

    @property
    def output_polydata(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        return self.__output_polydata


    def __divide(self):
        """ """

        print("TO-DO: DOCUMENTATION")
        print("TO-DO: COMPROBAR LV Y RV, QUE SEAN ESAS LAS PARTES DEL CLIPPING PLANE")

        # if self.anterior is None or self.posterior is None or self.apex is None:
        #     self.__calc_landmarks()

        clip = vtk.vtkClipPolyData()
        plane = vtk.vtkPlane()

        anterior    = self.anterior
        posterior   = self.posterior
        apex        = self.apex

        O = asarray(self.polydata.GetPoint(self.apex))
        A = asarray(self.polydata.GetPoint(self.anterior))
        B = asarray(self.polydata.GetPoint(self.posterior))

        OA = A - O
        OB = B - O

        # Clipping plane 1
        normal = cross(OA,OB)/norm(cross(OA,OB))

        plane.SetOrigin(self.polydata.GetPoint(self.apex))
        plane.SetNormal((normal[0], normal[1], normal[2]))

        clip.SetClipFunction(plane)
        clip.SetInputData(self.polydata)
        clip.Update()
        self.__RV_polydata = clip.GetOutput()

        # Clipping plane 2
        normal = cross(OB,OA)/norm(cross(OB,OA))

        plane.SetOrigin(self.polydata.GetPoint(self.apex))
        plane.SetNormal((normal[0], normal[1], normal[2]))

        clip.SetClipFunction(plane)
        clip.SetInputData(self.polydata)
        clip.Update()
        self.__LV_polydata = clip.GetOutput()


    def __calc_laplacian(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        numPoints           = self.polydata.GetNumberOfPoints()
        numPolygons         = self.polydata.GetNumberOfPolys()
        numDims             = self.polydata.GetPoints().GetData().GetNumberOfComponents()
        sparseMatrix        = csr_matrix((numPoints, numPoints))

        for i in range(0, numDims):
            i1              = (i + 0)%3
            i2              = (i + 1)%3
            i3              = (i + 2)%3

            vectP2P1        = (self.points[:, self.polygons[i2, :]] 
                              - self.points[:, self.polygons[i1, :]])
            vectP3P1        = (self.points[:, self.polygons[i3, :]] 
                              - self.points[:, self.polygons[i1, :]])

            vectP2P1        = vectP2P1 / repmat(sqrt((vectP2P1**2).sum(0)), 
                                                numDims, 1)
            vectP3P1        = vectP3P1 / repmat(sqrt((vectP3P1**2).sum(0)), 
                                                numDims, 1)

            angles          = arccos((vectP2P1 * vectP3P1).sum(0))

            iterData1       = csr_matrix((1/tan(angles), (self.polygons[i2,:], 
                                          self.polygons[i3,:])), 
                                         shape=(numPoints, numPoints))

            iterData2       = csr_matrix((1/tan(angles), (self.polygons[i3,:], 
                                          self.polygons[i2,:])), 
                                         shape=(numPoints, numPoints))

            sparseMatrix    = sparseMatrix + iterData1 + iterData2

        # diagonal            = sparseMatrix.sum(0)
        # diagonalSparse      = spdiags(diagonal, 0, numPoints, numPoints)
        # self.__laplacian    = diagonalSparse - sparseMatrix
        self.__laplacian    = sparseMatrix

    
    def __calc_homeomorphism(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        if (self.laplacian is not None) and (self.boundary is not None):
            numPoints               = self.polydata.GetNumberOfPoints()
            diagonal                = self.laplacian.sum(0)
            diagonalSparse          = spdiags(diagonal, 0, numPoints, numPoints)
            homeomorphism_laplacian = diagonalSparse - self.laplacian

            # Finds non-zero elements in the laplacian matrix
            (nzi, nzj)      = find(self.laplacian)[0:2]

            for point in self.boundary:
                positions   = where(nzi==point)[0]

                homeomorphism_laplacian[nzi[positions], nzj[positions]] = 0
                homeomorphism_laplacian[point, point] = 1

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

    def __calc_thin_plate_splines(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        if (self.homeomorphism is not None) and (self.apex is not None):
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

            result = thinPlateInterpolation(self.homeomorphism[0,:], self.homeomorphism[1,:])

            self.__homeomorphism[0,:] = result.real
            self.__homeomorphism[1,:] = result.imag

    def flip_boundary(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        self.__boundary         = flipud(self.boundary)
        self.__boundary         = roll(self.boundary, 1)




    ##########################################################################
    ##########################################################################
    def __closest_point(self, objectivePoint):
        """ """

        print("TO-DO: DOCUMENTATION")

        try:
            dimensions = len(anterior)
        except:
            dimensions = 0




        if objectivePoint == self.septum:
            return

        if objectivePoint in self.boundary:
            septalIndex         = objectivePoint
            closestPoint        = objectivePoint
            closestPointIndex   = where(self.boundary==objectivePoint)

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

        else:
            print(" *  Provided point not found in the boundary. Selecting \n"
                  "    closest point available...")

            try:
                searched_point  = self.points[:, objectivePoint]
            except:
                raise Exception("Septal point provided out of data bounds the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.")

            if len(self.boundary.shape) == 1:
                searched_point  = repmat(searched_point, self.boundary.size, 1)
                searched_point  = searched_point.transpose()
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.")

            distanceToObjectivePoint    = (self.points[:, self.boundary] - searched_point)
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0))
            closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

            septalIndex                 = self.boundary[closestPointIndex]

        self.__boundary                 = roll(self.boundary, -closestPointIndex)

        center  = asarray([0, 0]) # The center of the disk will always be a 
        vector1 = asarray([1, 0]) # point (0,0) and the septum a vector (1,0), 
                                   # as induced by the boundary conditions
        vector2 = self.homeomorphism[:, septalIndex] - center

        angle   = arccos(dot(vector1, vector2)/(norm(vector1)*norm(vector2)))

        # If the y coordinate of the vector w.r.t. the rotation will take place
        # is negative, the rotation must be done counterclock-wise
        if vector2[1] > 0:
            angle = -angle

        rotation_matrix                 = asarray([[cos(angle), -sin(angle)],
                                                   [sin(angle), cos(angle)]])

        self.__septum                   = septalIndex
        self.__homeomorphism               = rotation_matrix.dot(self.homeomorphism)

        self.__write_output()
    ##########################################################################
    ##########################################################################


    def __rearrange(self, objectivePoint=None):
        """ """

        print("TO-DO: DOCUMENTATION")

        septalIndex             = None
        septalPoint             = None
        closestPoint            = None

        if objectivePoint is None:
            if self.septum is None:
                raise Exception("No septal point provided in function call and no septal point provided in constructor. Aborting arrangement. ")
            else:
                septalIndex     = self.septum
        else:
            print(" *  Using provided septal point as rearranging point.")
            self.__septum       = objectivePoint
            septalIndex         = objectivePoint

        if septalIndex in self.boundary:
            closestPoint        = septalIndex
            closestPointIndex   = where(self.boundary==septalIndex)

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

            self.__boundary     = roll(self.boundary, -closestPointIndex)
        else:
            try:
                septalPoint     = self.points[:, septalIndex]
            except:
                raise Exception("Septal point provided out of data bounds the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.")

            if len(self.boundary.shape) == 1:
                septalPoint     = repmat(septalPoint,
                                    self.boundary.size, 1)
                septalPoint     = septalPoint .transpose()
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.")

            distanceToObjectivePoint    = (self.points[:, self.boundary] - septalPoint)
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0))
            closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

            self.__boundary     = roll(self.boundary, -closestPointIndex)

    # def rearrange_boundary(self, objectivePoint):
    #     """Rearranges the boundary aroung a new point identifier
    #     """
    #     old_septum              = self.septum
    #     septalIndex             = None
    #     closestPoint            = None

    #     if objectivePoint == self.septum:
    #         return

    #     if objectivePoint in self.boundary:
    #         septalIndex         = objectivePoint
    #         closestPoint        = objectivePoint
    #         closestPointIndex   = where(self.boundary==objectivePoint)

    #         if len(closestPointIndex) == 1:
    #             if len(closestPointIndex[0]) == 1:
    #                 closestPointIndex = closestPointIndex[0][0]
    #             else:
    #                 raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
    #         else:
    #             raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

    #     else:
    #         print(" *  Provided point not found in the boundary. Selecting \n"
    #               "    closest point available...")

    #         try:
    #             searched_point  = self.points[:, objectivePoint]
    #         except:
    #             raise Exception("Septal point provided out of data bounds the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.")

    #         if len(self.boundary.shape) == 1:
    #             searched_point  = repmat(searched_point, self.boundary.size, 1)
    #             searched_point  = searched_point.transpose()
    #         else:
    #             raise Exception("It seems you have multiple boundaries. Contact the package maintainer.")

    #         distanceToObjectivePoint    = (self.points[:, self.boundary] - searched_point)
    #         distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0))
    #         closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

    #         if len(closestPointIndex) == 1:
    #             if len(closestPointIndex[0]) == 1:
    #                 closestPointIndex   = closestPointIndex[0][0]
    #             else:
    #                 raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
    #         else:
    #             raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

    #         septalIndex                 = self.boundary[closestPointIndex]

    #     self.__boundary                 = roll(self.boundary, -closestPointIndex)

    #     center  = asarray([0, 0]) # The center of the disk will always be a 
    #     vector1 = asarray([1, 0]) # point (0,0) and the septum a vector (1,0), 
    #                                # as induced by the boundary conditions
    #     vector2 = self.homeomorphism[:, septalIndex] - center

    #     angle   = arccos(dot(vector1, vector2)/(norm(vector1)*norm(vector2)))

    #     # If the y coordinate of the vector w.r.t. the rotation will take place
    #     # is negative, the rotation must be done counterclock-wise
    #     if vector2[1] > 0:
    #         angle = -angle

    #     rotation_matrix                 = asarray([[cos(angle), -sin(angle)],
    #                                                [sin(angle), cos(angle)]])

    #     self.__septum                   = septalIndex
    #     self.__homeomorphism               = rotation_matrix.dot(self.homeomorphism)

    #     self.__write_output()


    def closest_boundary_point(self, objectivePoint=None):
        """ """

        print("TO-DO: DOCUMENTATION")

        if objectivePoint is None:
            selected = True
            
            while(selected):
                ps = PointSelector()
                ps.DoSelection(self.polydata)

                if ps.GetSelectedPoints().GetNumberOfPoints() == 1:
                    selected = False

                    objectivePoint  = ps.GetSelectedPointIds().GetId(0)


        output_point = None

        if objectivePoint in self.boundary:
            output_point        = where(self.boundary==objectivePoint)

            if len(output_point) == 1:
                if len(output_point[0]) == 1:
                    output_point = output_point[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

        else:
            print(" *  Provided point not found in the boundary. Selecting \n"
                  "    closest point available...")

            try:
                searched_point  = self.points[:, objectivePoint]
            except:
                raise Exception("Septal point provided out of data bounds the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.")

            if len(self.boundary.shape) == 1:
                searched_point  = repmat(searched_point, self.boundary.size, 1)
                searched_point  = searched_point.transpose()
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.")

            distanceToObjectivePoint    = (self.points[:, self.boundary] - searched_point)
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0))
            output_point                = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

            if len(output_point) == 1:
                if len(output_point[0]) == 1:
                    output_point   = output_point[0][0]
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.")

        return self.boundary[output_point]


    def __write_output(self):
        """ """

        print("TO-DO: DOCUMENTATION")

        if ((self.homeomorphism is not None)
            and (self.points is not None)
            and (self.polygons is not None)):

            newPolyData         = vtkPolyData()
            newPointData        = vtkPoints()
            writer              = vtkPolyDataWriter()

            if self.output_path is self.path:
                print(" *  Writing to default location: ")

                path                = None

                directory, filename = split(self.path)
                filename, extension = splitext(filename)

                if isdir(join(directory, 'QCM')):
                    path            = join(directory, 'QCM', str(filename + '_QCM' + extension))
                else:
                    mkdir(join(directory, 'QCM'))

                    if isdir(join(directory, 'QCM')):
                        path        = join(directory, 'QCM', str(filename + '_QCM' + extension))
                    else:
                        path        = join(directory, str(filename + '_QCM' + extension))

                print("    " + path + "\n")

            else:
                if splitext(self.output_path)[1] is '':
                    self.__output_path  = self.output_path + ".vtk"

                path                    = self.output_path

            self.__output_path          = path
            writer.SetFileName(path)

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

            # In case the host computer converts decimal points (.) to decimal
            # commas (,), such as in Spanish locales.
            system("perl -pi -e 's/,/./g' %s " % path)

        else:
            raise RuntimeError("Information provided insufficient")

    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path + "'.\n"
        s = s + "Number of dimensions: " + str(self.polydata.GetPoints().GetData().GetNumberOfComponents()) + "\n"
        s = s + "Number of points: " + str(self.polydata.GetNumberOfPoints()) + "\n"
        s = s + "Number of polygons: " + str(self.polydata.GetNumberOfCells()) + "\n"
        s = s + "Output file location: " + str(self.output_path)
        return s