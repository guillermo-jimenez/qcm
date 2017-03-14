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
import mvpoly.rbf;
import vtk;

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
# from scipy.interpolate import Rbf;
# from mvpoly.rbf import RBFThinPlateSpline;

from scipy.sparse import csr_matrix;
from scipy.sparse import spdiags;
from scipy.sparse.linalg import spsolve; 


# import time;

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
    __boundary              = None;
    __output                = None;

    def __init__(self, path, septum, apex):
        if os.path.isfile(path):
            self.__path     = path;
        else:
            raise RuntimeError("File does not exist.");
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
        self.__CalculateThinPlateSplines();

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

    def __CalculateThinPlateSplines(self):
        if self.__output is not None:
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

                x = source[0,:];
                y = source[1,:];
                d = destination[0,:] + 1j*destination[1,:];

                thinPlateInterpolation = mvpoly.rbf.RBFThinPlateSpline(x,y,d);
                result = thinPlateInterpolation(self.__output[0,:], self.__output[1,:]);

                self.__output[0,:] = result.real;
                self.__output[1,:] = result.imag;

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

# septum_MRI = 201479 - 1;
# apex_MRI = 37963 - 1;
# path_MRI = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

# apex_EAM = 599 - 1;
# septum_EAM = 1389 - 1;
# path_EAM = os.path.join("/home/guille/BitBucket/qcm/data/pat1/EAM", "pat1_EAM_endo_smooth.vtk");

# start = time.time(); reload(VentricularImage); MRI = VentricularImage.VentricularImage(path_MRI, septum_MRI, apex_MRI); print(time.time() - start);
# start = time.time(); reload(VentricularImage); EAM = VentricularImage.VentricularImage(path_EAM, septum_EAM, apex_EAM); print(time.time() - start);



