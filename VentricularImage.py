"""
    Copyright (C) 2017 - Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>

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



# for subdir, dirs, files in os.walk(rootdir):
# for file in files:


# def docstring_example():
# """ 
# Multi-line Docstrings

# Multi-line docstrings consist of a summary line just like a one-line
# docstring, followed by a blank line, followed by a more elaborate
# description. The summary line may be used by automatic indexing tools;
# it is important that it fits on one line and is separated from the rest
# of the docstring by a blank line. The summary line may be on the same
# line as the opening quotes or on the next line. The entire docstring is
# indented the same as the quotes at its first line (see example below).

# Example:
# def complex(real=0.0, imag=0.0):
# '''
# Form a complex number.

# Keyword arguments:
# real -- the real part (default 0.0)
# imag -- the imaginary part (default 0.0)
# ...
# '''


# """


from __future__ import division

import os;
import numpy;
import scipy;

from numpy.matlib import repmat;
from numpy import int;
from scipy import zeros;
from scipy import asarray;
from scipy import mean;
from scipy import sqrt;
from scipy import pi;
from scipy import sin;
from scipy import cos;
from scipy import tan;
from scipy import arccos;
from scipy import cumsum;
from scipy import where;
from scipy import roll;
from scipy import flip;
from scipy.interpolate import Rbf;

from scipy.sparse import csr_matrix;
from scipy.sparse import spdiags;
from scipy.sparse.linalg import spsolve; 

# import time;
import vtk;

class VentricularImage(object):
    """ DOCSTRING """

    __imageType             = None;
    __path                  = None;
    __originalPolyData      = None;
    # __QCMPolyData           = None;
    __pointData             = None;
    __polygonData           = None;
    __scalarData            = None;
    # __normalData            = None;
    __nPoints               = None;
    __nPolygons             = None;
    __septum                = None;
    __apex                  = None;
    __laplacianMatrix       = None;
    # __constrain             = None;
    __boundary              = None;
    __output                = None;

    def __init__(self, path, septum, apex):
        if os.path.isfile(path):
            self.__path     = path;
        else:
            raise RuntimeError("File does not exist.");
        # self.__path         = path;
        self.__ReadPolyData();
        self.__ReadPointData();
        self.__ReadPolygonData();
        self.__ReadNormalData();
        self.__ReadScalarData();
        self.__septum       = septum;
        self.__apex         = apex;
        self.__nPoints      = self.__pointData.shape[1];
        self.__nPolygons    = self.__polygonData.shape[1];
        self.__CalculateBoundary();
        self.RearrangeBoundary();
        self.__LaplacianMatrix();
        self.__CalculateLinearTransformation();

        # if path is None:
        #     if septum is not None:
        #         self.__septum   = septum;
        # else:
        #     if os.path.isfile(path):
        #         self.__path     = path;
        #     else:
        #         raise RuntimeError("File does not exist.");

        #     # start               = time.time();
        #     self.__ReadPolyData();
        #     self.__ReadPointData();
        #     self.__ReadPolygonData();
        #     self.__ReadNormalData();
        #     self.__ReadScalarData();
        #     self.__septum       = septum;
        #     self.__nPoints      = self.__pointData.shape[1];
        #     self.__nPolygons    = self.__polygonData.shape[1];
        #     self.__CalculateBoundary();
        #     self.RearrangeBoundary();
        #     # start = time.time();
        #     self.__LaplacianMatrix();
        #     # print(time.time()-start);
        #     self.__CalculateLinearTransformation();

    def __ReadPolyData(self):
        reader                  = vtk.vtkPolyDataReader();
        reader.SetFileName(self.__path);
        reader.Update();

        polyData                = reader.GetOutput();
        polyData.BuildLinks();

        self.__originalPolyData = polyData;

    def __ReadPolygonData(self):
        rows                    = None;
        cols                    = None;
        polygons                = None;

        polys                   = self.__originalPolyData.GetPolys();

        for i in xrange(self.__originalPolyData.GetNumberOfCells()):
            triangle            = self.__originalPolyData.GetCell(i);
            pointIds            = triangle.GetPointIds();

            if polygons is None:
                rows            = pointIds.GetNumberOfIds();
                cols            = self.__originalPolyData.GetNumberOfCells();
                polygons        = scipy.zeros((rows,cols), dtype=numpy.int);
            
            polygons[0,i]       = pointIds.GetId(0);
            polygons[1,i]       = pointIds.GetId(1);
            polygons[2,i]       = pointIds.GetId(2);

        self.__polygonData      = polygons;

    def __ReadPointData(self):
        rows                    = None;
        cols                    = None;
        points                  = None;

        pointVector             = self.__originalPolyData.GetPoints();

        if pointVector:
            for i in range(0, pointVector.GetNumberOfPoints()):
                point_tuple     = pointVector.GetPoint(i);

                if points is None:
                    rows        = len(point_tuple);
                    cols        = pointVector.GetNumberOfPoints();
                    points      = scipy.zeros((rows,cols));
                
                points[0,i]     = point_tuple[0];
                points[1,i]     = point_tuple[1];
                points[2,i]     = point_tuple[2];

        self.__pointData        = points;

    def __ReadNormalData(self):
        rows                    = None;
        cols                    = None;
        normals                 = None;

        normalVector            = self.__originalPolyData.GetPointData().GetNormals();

        if normalVector:
            for i in range(0, normalVector.GetNumberOfTuples()):
                normalTuple     = normalVector.GetTuple(i);

                if normals is None:
                    rows        = len(normalTuple);
                    cols        = normalVector.GetNumberOfTuples();
                    normals     = scipy.zeros((rows,cols));
                
                normals[0,i]    = normalTuple[0];
                normals[1,i]    = normalTuple[1];
                normals[2,i]    = normalTuple[2];

        self.__normalData       = normals;

    def __ReadScalarData(self):
        rows                    = None;
        cols                    = None;
        scalars                 = None;

        scalarVector            = self.__originalPolyData.GetPointData().GetScalars();

        if scalarVector:
            for i in xrange(scalarVector.GetNumberOfTuples()):
                scalarTuple     = scalarVector.GetTuple(i);

                if scalars is None:
                    rows        = len(scalarTuple);
                    cols        = scalarVector.GetNumberOfTuples();
                    scalars     = scipy.zeros((rows,cols));
                
                for j in xrange(len(scalarTuple)):
                    scalars[j,i] = scalarTuple[j];

        self.__scalarData       = scalars;

    def __LaplacianMatrix(self):
        numDims                 = self.__polygonData.shape[0];
        numPoints               = self.__pointData.shape[1];
        numPolygons             = self.__polygonData.shape[1];
        boundary                = self.__boundary;
        boundaryConstrain       = scipy.zeros((2,numPoints));

        sparseMatrix            = scipy.sparse.csr_matrix((numPoints, numPoints));

        for i in range(0, numDims):
            i1                  = (i + 0)%3;
            i2                  = (i + 1)%3;
            i3                  = (i + 2)%3;

            distP2P1            = self.__pointData[:, self.__polygonData[i2, :]] - self.__pointData[:, self.__polygonData[i1, :]];
            distP3P1            = self.__pointData[:, self.__polygonData[i3, :]] - self.__pointData[:, self.__polygonData[i1, :]];

            distP2P1            = distP2P1 / numpy.matlib.repmat(scipy.sqrt((distP2P1**2).sum(0)), 3, 1);
            distP3P1            = distP3P1 / numpy.matlib.repmat(scipy.sqrt((distP3P1**2).sum(0)), 3, 1);

            angles              = scipy.arccos((distP2P1 * distP3P1).sum(0));

            iterData1           = scipy.sparse.csr_matrix((1/scipy.tan(angles), 
                                                    (self.__polygonData[i2,:], 
                                                     self.__polygonData[i3,:])), 
                                                    shape=(numPoints, numPoints));

            iterData2           = scipy.sparse.csr_matrix((1/scipy.tan(angles), (self.__polygonData[i3,:], self.__polygonData[i2,:])), shape=(numPoints, numPoints));

            sparseMatrix        = sparseMatrix + iterData1 + iterData2;

        diagonal                = sparseMatrix.sum(0);
        diagonalSparse          = scipy.sparse.spdiags(diagonal, 0, numPoints, numPoints);
        self.__laplacianMatrix  = diagonalSparse - sparseMatrix;

    def __CalculateLinearTransformation(self):
        if self.__laplacianMatrix is not None:
            if self.__boundary is not None:
                laplacian       = self.__laplacianMatrix;
                (nzi, nzj)      = scipy.sparse.find(laplacian)[0:2];

                for point in self.__boundary:
                    positions   = scipy.where(nzi==point)[0];

                    laplacian[nzi[positions], nzj[positions]] = 0;

                    laplacian[point, point] = 1;

                Z = self.GetWithinBoundarySinCos();

                boundaryConstrain = scipy.zeros((2, self.__nPoints));
                boundaryConstrain[:, self.__boundary] = Z;

                self.__output   = scipy.sparse.linalg.spsolve(laplacian, boundaryConstrain.transpose()).transpose();

    def CalculateThinPlateSplines(self):
        if self.__output is not None:
            # thinPlate           = scipy.interpolate.Rbf()

            if self.__apex is not None:
                boundaryPoints  = self.__output[:,self.__boundary];
                source          = scipy.zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));
                destination     = scipy.zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));

                source[:, 0:source.shape[1] - 1]        = boundaryPoints;
                source[:, source.shape[1] - 1]          = self.__output[:, self.__apex];

                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;
                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;

                if self.__apex in self.__boundary:
                    closestPoint            = self.__apex;
                    closestBoundaryToApex   = scipy.where(self.__boundary==closestPoint);

                    if len(closestBoundaryToApex) == 1:
                        if len(closestBoundaryToApex[0]) == 1:
                            closestBoundaryToApex = closestBoundaryToApex[0][0];
                        else:
                            raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
                    else:
                        raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

                    # self.__boundary     = scipy.roll(self.__boundary, -closestBoundaryToApex);
                else:
                    try:
                        apexPoint       = self.__output[:, self.__apex];
                    except:
                        raise Exception("Septal point provided out of data bounds; the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.");

                    if len(self.__boundary.shape) == 1:
                        apexPoint       = numpy.matlib.repmat(apexPoint,
                                            self.__boundary.size, 1);
                        apexPoint       = apexPoint.transpose();
                    else:
                        raise Exception("It seems you have multiple boundaries. Contact the package maintainer.");

                    distanceToObjectivePoint    = self.__output[:, self.__boundary] - apexPoint;
                    distanceToObjectivePoint    = scipy.sqrt((distanceToObjectivePoint**2).sum(0));
                    closestBoundaryToApex       = scipy.where(distanceToObjectivePoint == distanceToObjectivePoint.min());

                    if len(closestBoundaryToApex) == 1:
                        if len(closestBoundaryToApex[0]) == 1:
                            closestBoundaryToApex   = closestBoundaryToApex[0][0];
                        else:
                            raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
                    else:
                        raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

                return source, destination;

                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

                # thinPlateSplines        = scipy.interpolate.Rbf(source, destination, 1, function='thin-plate');

                # return thinPlateSplines(self.__output);

    def __CalculateBoundary(self):
        startingPoint           = None;
        currentPoint            = None;
        foundBoundary           = False;
        cellId                  = None;
        boundary                = [];
        visitedEdges            = [];
        visitedPoints           = [];
        visitedBoundaryEdges    = [];

        for cellId in xrange(self.__originalPolyData.GetNumberOfCells()):
            cellPointIdList     = vtk.vtkIdList();
            cellEdges           = [];

            self.__originalPolyData.GetCellPoints(cellId, cellPointIdList);

            cellEdges           = [[cellPointIdList.GetId(0), 
                                    cellPointIdList.GetId(1)], 
                                   [cellPointIdList.GetId(1), 
                                    cellPointIdList.GetId(2)], 
                                   [cellPointIdList.GetId(2), 
                                    cellPointIdList.GetId(0)]];

            for i in xrange(len(cellEdges)):
                if (cellEdges[i] in visitedEdges) == False:
                    visitedEdges.append(cellEdges[i]);

                    edgeIdList  = vtk.vtkIdList()
                    edgeIdList.InsertNextId(cellEdges[i][0]);
                    edgeIdList.InsertNextId(cellEdges[i][1]);

                    singleCellEdgeNeighborIds = vtk.vtkIdList();

                    self.__originalPolyData.GetCellEdgeNeighbors(cellId, cellEdges[i][0], cellEdges[i][1], singleCellEdgeNeighborIds);

                    if singleCellEdgeNeighborIds.GetNumberOfIds() == 0:
                        foundBoundary   = True;

                        startingPoint   = cellEdges[i][0];
                        currentPoint    = cellEdges[i][1];

                        boundary.append(cellEdges[i][0]);
                        boundary.append(cellEdges[i][1]);

                        visitedBoundaryEdges.append([currentPoint,startingPoint]);
                        visitedBoundaryEdges.append([startingPoint,currentPoint]);

            if foundBoundary == True:
                break;

        if foundBoundary == False:
            raise Exception("The mesh provided has no boundary; not possible to do Quasi-Conformal Mapping on this dataset.");

        while currentPoint != startingPoint:
            neighboringCells    = vtk.vtkIdList();

            self.__originalPolyData.GetPointCells(currentPoint, neighboringCells);

            for i in xrange(neighboringCells.GetNumberOfIds()):
                cell = neighboringCells.GetId(i);
                triangle = self.__originalPolyData.GetCell(cell);

                for j in xrange(triangle.GetNumberOfPoints()):
                    if triangle.GetPointId(j) == currentPoint:
                        j1      = (j + 1) % 3;
                        j2      = (j + 2) % 3;

                        edge1   = [triangle.GetPointId(j),
                             triangle.GetPointId(j1)];
                        edge2   = [triangle.GetPointId(j),
                             triangle.GetPointId(j2)];

                edgeNeighbors1  = vtk.vtkIdList();
                edgeNeighbors2  = vtk.vtkIdList();

                self.__originalPolyData.GetCellEdgeNeighbors(cell, edge1[0], edge1[1], edgeNeighbors1);

                self.__originalPolyData.GetCellEdgeNeighbors(cell, edge2[0], edge2[1], edgeNeighbors2);

                if edgeNeighbors1.GetNumberOfIds() == 0:
                    if ([edge1[1], edge1[0]] in visitedBoundaryEdges) == False:
                        if (edge1[1] in boundary) == False:
                            boundary.append(edge1[1]);
                        visitedBoundaryEdges.append([edge1[0], edge1[1]]);
                        visitedBoundaryEdges.append([edge1[1], edge1[0]]);
                        currentPoint = edge1[1];
                        break;

                if edgeNeighbors2.GetNumberOfIds() == 0:
                    if ([edge2[1], edge2[0]] in visitedBoundaryEdges) == False:
                        if (edge2[1] in boundary) == False:
                            boundary.append(edge2[1]);
                        visitedBoundaryEdges.append([edge2[0], edge2[1]]);
                        visitedBoundaryEdges.append([edge2[1], edge2[0]]);
                        currentPoint = edge2[1];
                        break;

        boundary    = scipy.asarray(boundary, dtype=int);

        center      = scipy.mean(self.__pointData[:,boundary], axis=1);
        # vector1     = scipy.asarray()
        vector1     = scipy.asarray(self.__pointData[:,boundary[0]] - center);
        vector2     = scipy.asarray(self.__pointData[:,boundary[1]] - center);
        vectorNormal= scipy.cross(vector1, vector2);
        vectorApex  = self.__pointData[:, self.__apex] - center;

        if len(center.shape) is not 1:
            if center.shape[0] is not 3:
                raise Exception("Something went wrong. Probably forgot to transpose this. Contact maintainer.");

        if scipy.dot(vectorApex, vectorNormal) < 0:
            boundary         = scipy.flip(boundary, 0);
            boundary         = scipy.roll(boundary, 1);

        self.__boundary = boundary;

    def GetPolyData(self):
        return self.__originalPolyData;

    def GetPointData(self):
        return self.__pointData;

    def GetImageType(self):
        return self.__imageType;

    def GetPath(self):
        return self.__path;

    def GetNormalData(self):
        return self.__normalData;

    def GetScalarData(self):
        return self.__scalarData;

    def GetPolygonData(self):
        return self.__polygonData;

    def GetNumberOfPoints(self):
        return self.__nPoints;

    def GetNumberOfPolygons(self):
        return self.__nPolygons;

    def GetSeptumId(self):
        return self.__septum;

    def GetLaplacianMatrix(self):
        return self.__laplacianMatrix;

    def GetBoundary(self):
        return self.__boundary;

    def GetOutput(self):
        return self.__output;

    def GetBoundaryPoints(self):
        return self.__pointData[:, self.__boundary];

    def FlipBoundary(self):
        self.__boundary         = scipy.flip(self.__boundary, 0);
        self.__boundary         = scipy.roll(self.__boundary, 1);

    def GetWithinBoundaryDistances(self):
        boundaryNext            = scipy.roll(self.__boundary, -1);
        boundaryNextPoints      = self.__pointData[:, boundaryNext];

        distanceToNext          = boundaryNextPoints - self.GetBoundaryPoints();

        return scipy.sqrt((distanceToNext**2).sum(0));

    def GetPerimeter(self):
        return self.GetWithinBoundaryDistances().sum();

    def GetWithinBoundaryDistancesAsFraction(self):
        euclideanNorm           = self.GetWithinBoundaryDistances();
        perimeter               = euclideanNorm.sum();

        return euclideanNorm/perimeter;

    def GetWithinBoundaryAngles(self):
        circleLength            = 2*scipy.pi;
        fraction                = self.GetWithinBoundaryDistancesAsFraction();

        angles                  = scipy.cumsum(circleLength*fraction);
        angles                  = scipy.roll(angles, 1);
        angles[0]               = 0;

        return angles;

    def GetWithinBoundarySinCos(self):
        angles                  = self.GetWithinBoundaryAngles();
        Z                       = scipy.zeros((2, angles.size));
        Z[0,:]                  = scipy.cos(angles);
        Z[1,:]                  = scipy.sin(angles);

        return Z;

    def RearrangeBoundary(self, objectivePoint=None):
        septalIndex             = None;
        septalPoint             = None;
        closestPoint            = None;

        if objectivePoint is None:
            if self.__septum is None:
                raise Exception("No septal point provided in function call and no septal point provided in constructor. Aborting arrangement. ");
            else:
                septalIndex     = self.__septum;
        else:
            print("Using provided septal point as rearranging point.");
            self.__septum       = objectivePoint;
            septalIndex         = objectivePoint;

        if septalIndex in self.__boundary:
            closestPoint        = septalIndex;
            closestPointIndex   = scipy.where(self.__boundary==septalIndex);

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary     = scipy.roll(self.__boundary, -closestPointIndex);
        else:
            try:
                septalPoint     = self.__pointData[:, septalIndex];
            except:
                raise Exception("Septal point provided out of data bounds; the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.");

            if len(self.__boundary.shape) == 1:
                septalPoint     = numpy.matlib.repmat(septalPoint,
                                    self.__boundary.size, 1);
                septalPoint     = septalPoint .transpose();
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.");

            distanceToObjectivePoint    = (self.__pointData[:, self.__boundary] - septalPoint);
            distanceToObjectivePoint    = scipy.sqrt((distanceToObjectivePoint**2).sum(0));
            closestPointIndex           = scipy.where(distanceToObjectivePoint == distanceToObjectivePoint.min());
            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary             = scipy.roll(self.__boundary, 
                                          -closestPointIndex);







# import scipy, time, os, numpy, VentricularImage;

# septum = 201479 - 1;
# apex = 37963 - 1;

# path = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
# start = time.time(); reload(VentricularImage); MRI = VentricularImage.VentricularImage(path, septum, apex); print(time.time() - start);




# start = time.time(); MRI2 = VentricularImage(path); print(time.time() - start);


# MRI.CalculateLinearTransformation();

# A = MRI.CalculateLinearTransformation();




# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA


source, destination     = MRI.CalculateThinPlateSplines();

x           = source.transpose();
y           = destination.transpose();

A           = scipy.zeros((x.shape[0], 3));
A[:, :2]    = x;
A[:, 2]     = 1;

Q, R        = scipy.linalg.qr(A)
radiags     = scipy.sort(scipy.absolute(scipy.diagonal(R)))

Q1          = Q[:,0:3]; 
Q           = scipy.delete(Q,[0,1,2], axis=1);

# # form        = 'st-tp'; centers = x; coefs = []; interv = {[],[]};

B           = scipy.asmatrix(y.transpose()[1,:])*Q;






















# import numpy as np
# import scipy.sparse.linalg as spla
 
# A = np.array([[ 0.4445,  0.4444, -0.2222],
#               [ 0.4444,  0.4445, -0.2222],
#               [-0.2222, -0.2222,  0.1112]])

# b = np.array([[ 0.6667], 
#               [ 0.6667], 
#               [-0.3332]])

# M2 = spla.spilu(A)
# M_x = lambda x: M2.solve(x)
# M = spla.LinearOperator((3,3), M_x)

# x = spla.gmres(A,b,M=M)

# print x







################################################################################
# Using gmres with a Function Handle

# This example replaces the matrix A in the previous example with a handle to a
# matrix-vector product function afun, and the preconditioner M1 with a handle
# to a backsolve function mfun. The example is contained in a function run_gmres
# that

# Calls gmres with the function handle @afun as its first argument.
# Contains afun and mfun as nested functions, so that all variables in run_gmres
# are available to afun and mfun.
# The following shows the code for run_gmres:

# function x1 = run_gmres
# n = 21;
# b = afun(ones(n,1));
# tol = 1e-12;  maxit = 15; 
# x1 = gmres(@afun,b,10,tol,maxit,@mfun);
 
#     function y = afun(x)
#         y = [0; x(1:n-1)] + ...
#               [((n-1)/2:-1:0)'; (1:(n-1)/2)'].*x + ...
#               [x(2:n); 0];
#     end
 
#     function y = mfun(r)
#         y = r ./ [((n-1)/2:-1:1)'; 1; (1:(n-1)/2)'];
#     end
# end
# When you enter

# x1 = run_gmres;
# MATLAB software displays the message

# gmres(10) converged at outer iteration 2 (inner iteration 10) 
# to a solution with relative residual 1.1e-013.
# Using a Preconditioner without Restart
# This example demonstrates the use of a preconditioner without restarting gmres.

# Load west0479, a real 479-by-479 nonsymmetric sparse matrix.

# load west0479;
# A = west0479;
# Set the tolerance and maximum number of iterations.

# tol = 1e-12;
# maxit = 20;
# Define b so that the true solution is a vector of all ones.

# b = full(sum(A,2));
# [x0,fl0,rr0,it0,rv0] = gmres(A,b,[],tol,maxit);

# fl0 is 1 because gmres does not converge to the requested tolerance 1e-12
# within the requested 20 iterations. The best approximate solution that gmres
# returns is the last one (as indicated by it0(2) = 20). MATLAB stores the
# residual history in rv0.

# Plot the behavior of gmres.

# semilogy(0:maxit,rv0/norm(b),'-o');
# xlabel('Iteration number');
# ylabel('Relative residual');


# The plot shows that the solution converges slowly. A preconditioner may improve the outcome.

# Use ilu to form the preconditioner, since A is nonsymmetric.

# [L,U] = ilu(A,struct('type','ilutp','droptol',1e-5));
# Error using ilu
# There is a pivot equal to zero. Consider decreasing
# the drop tolerance or consider using the 'udiag' option.
# Note MATLAB cannot construct the incomplete LU as it would result in a
# singular factor, which is useless as a preconditioner.

# As indicated by the error message, try again with a reduced drop tolerance.

# [L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));
# [x1,fl1,rr1,it1,rv1] = gmres(A,b,[],tol,maxit,L,U);

# fl1 is 0 because gmres drives the relative residual to 9.5436e-14 (the value
# of rr1). The relative residual is less than the prescribed tolerance of 1e-12
# at the sixth iteration (the value of it1(2)) when preconditioned by the
# incomplete LU factorization with a drop tolerance of 1e-6. The output, rv1(1),
# is norm(M\b), where M = L*U. The output, rv1(7), is norm(U\(L\(b-A*x1))).

# Follow the progress of gmres by plotting the relative residuals at each
# iteration starting from the initial estimate (iterate number 0).

# semilogy(0:it1(2),rv1/norm(b),'-o');
# xlabel('Iteration number');
# ylabel('Relative residual');


# Using a Preconditioner with Restart
# This example demonstrates the use of a preconditioner with restarted gmres.

# Load west0479, a real 479-by-479 nonsymmetric sparse matrix.

# load west0479;
# A = west0479;
# Define b so that the true solution is a vector of all ones.

# b = full(sum(A,2));
# Construct an incomplete LU preconditioner as in the previous example.

# [L,U] = ilu(A,struct('type','ilutp','droptol',1e-6));

# The benefit to using restarted gmres is to limit the amount of memory required
# to execute the method. Without restart, gmres requires maxit vectors of
# storage to keep the basis of the Krylov subspace. Also, gmres must
# orthogonalize against all of the previous vectors at each step. Restarting
# limits the amount of workspace used and the amount of work done per outer
# iteration. Note that even though preconditioned gmres converged in six
# iterations above, the algorithm allowed for as many as twenty basis vectors
# and therefore, allocated all of that space up front.

# Execute gmres(3), gmres(4), and gmres(5)

# tol = 1e-12;
# maxit = 20;
# re3 = 3;
# [x3,fl3,rr3,it3,rv3] = gmres(A,b,re3,tol,maxit,L,U);
# re4 = 4;
# [x4,fl4,rr4,it4,rv4] = gmres(A,b,re4,tol,maxit,L,U);
# re5 = 5;
# [x5,fl5,rr5,it5,rv5] = gmres(A,b,re5,tol,maxit,L,U);

# fl3, fl4, and fl5 are all 0 because in each case restarted gmres drives the
# relative residual to less than the prescribed tolerance of 1e-12.

# The following plots show the convergence histories of each restarted gmres
# method. gmres(3) converges at outer iteration 5, inner iteration 3 (it3 = [5,
# 3]) which would be the same as outer iteration 6, inner iteration 0, hence the
# marking of 6 on the final tick mark.

# figure
# semilogy(1:1/3:6,rv3/norm(b),'-o');
# h1 = gca;
# h1.XTick = [1:1/3:6];
# h1.XTickLabel = ['1';' ';' ';'2';' ';' ';'3';' ';' ';'4';' ';' ';'5';' ';' ';'6';];
# title('gmres(3)')
# xlabel('Iteration number');
# ylabel('Relative residual');

# figure
# semilogy(1:1/4:3,rv4/norm(b),'-o');
# h2 = gca;
# h2.XTick = [1:1/4:3];
# h2.XTickLabel = ['1';' ';' ';' ';'2';' ';' ';' ';'3'];
# title('gmres(4)')
# xlabel('Iteration number');
# ylabel('Relative residual');

# figure
# semilogy(1:1/5:2.8,rv5/norm(b),'-o');
# h3 = gca;
# h3.XTick = [1:1/5:2.8];
# h3.XTickLabel = ['1';' ';' ';' ';' ';'2';' ';' ';' ';' '];
# title('gmres(5)')
# xlabel('Iteration number');
# ylabel('Relative residual');







