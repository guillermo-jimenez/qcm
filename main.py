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



# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:


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


from __future__ import division

import os;
import numpy;
import scipy;

from numpy.matlib import repmat;
from numpy import int;
from scipy import roll;
from scipy import flip;
from scipy import zeros;
from scipy import arccos;
from scipy import sin;
from scipy import cos;
from scipy import sqrt;
from scipy import asarray;
from scipy import tan;
from scipy import pi;
from scipy import cumsum;
from scipy import where;

from scipy.sparse import csr_matrix;
from scipy.sparse import spdiags;
from scipy.sparse.linalg import spsolve; 

import time;
import vtk;


start = time.time();

class VentricularImage:
    """ DOCSTRING """

    imageType                       = None;

    path                            = None;
    originalPolyData                = None;
    QCMPolyData                     = None;

    pointData                       = None;
    polygonData                     = None;

    scalarData                      = None;
    normalData                      = None;

    __nPoints                       = None;
    __nPolygons                     = None;
    __nDims                         = None;

    septum                          = None;

    laplacianMatrix                 = None;
    linearMatrix                    = None;

    boundary                        = None;

    def __init__(self, path=None, septum=None):
        if path is None:
            if septum is not None:
                self.septum         = septum;
        else:
            if os.path.isfile(path):
                self.path           = path;
            else:
                raise RuntimeError("File does not exist. If file exists and this"  \
                                   + "error was raised when automatically "        \
                                   + "scanning documents in a folder, contact "    \
                                   + "the package maintainer.");

            self.__ReadPolyData();
            self.__ReadPointData();
            self.__ReadPolygonData();
            self.__ReadNormalData();
            self.__ReadScalarData();
            self.septum             = septum;
            self.__nPoints          = self.pointData.shape[1];
            self.__nPolygons        = self.polygonData.shape[1];
            self.__CalculateBoundary();
            self.RearrangeBoundary();
            self.__LaplacianMatrix();

    def __ReadPolyData(self):
        reader                      = vtk.vtkPolyDataReader();
        reader.SetFileName(self.path);
        reader.Update();

        polyData                    = reader.GetOutput();
        polyData.BuildLinks();

        self.originalPolyData       = polyData;

    def __ReadPolygonData(self):
        rows                        = None;
        cols                        = None;
        polygons                    = None;

        try:
            polys = self.originalPolyData.GetPolys();
        except:
            raise RuntimeError("Tried to call function 'extract_polygons' with a " \
                               + "variable with no 'GetPolys()' method.");

        for i in xrange(self.originalPolyData.GetNumberOfCells()):
            triangle                = self.originalPolyData.GetCell(i);
            pointIds                = triangle.GetPointIds();

            if polygons is None:
                rows                = pointIds.GetNumberOfIds();
                cols                = self.originalPolyData.GetNumberOfCells();
                polygons            = scipy.zeros((rows,cols), dtype=numpy.int);
            
            polygons[0,i]           = pointIds.GetId(0);
            polygons[1,i]           = pointIds.GetId(1);
            polygons[2,i]           = pointIds.GetId(2);

        self.polygonData            = polygons;

    def __ReadPointData(self):
        rows                        = None;
        cols                        = None;
        points                      = None;

        try:
            pointVector             = self.originalPolyData.GetPoints();
        except:
            raise RuntimeError("Tried to call function 'extract_points' with a "   \
                               + "variable with no 'GetPoints()' method.");

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

        self.pointData              = points;

    def __ReadNormalData(self):
        rows                        = None;
        cols                        = None;
        normals                     = None;

        try:
            normalVector            = self.originalPolyData.GetPointData().GetNormals();
        except:
            raise RuntimeError("Tried to call function 'extract_normals' with a "  \
                               + "variable with no 'GetNormals()' method.");

        if normalVector:
            for i in range(0, normalVector.GetNumberOfTuples()):
                normalTuple         = normalVector.GetTuple(i);

                if normals is None:
                    rows            = len(normalTuple);
                    cols            = normalVector.GetNumberOfTuples();
                    normals         = scipy.zeros((rows,cols));
                
                normals[0,i]        = normalTuple[0];
                normals[1,i]        = normalTuple[1];
                normals[2,i]        = normalTuple[2];

        self.normalData             = normals;

    def __ReadScalarData(self):
        rows                        = None;
        cols                        = None;
        scalars                     = None;

        try:
            scalarVector            = self.originalPolyData.GetPointData().GetScalars();
        except:
            raise RuntimeError("Tried to call function 'extract_normals' with a "  \
                               + "variable with no 'GetScalars()' method.");

        if scalarVector:
            for i in xrange(scalarVector.GetNumberOfTuples()):
                scalarTuple         = scalarVector.GetTuple(i);

                if scalars is None:
                    rows            = len(scalarTuple);
                    cols            = scalarVector.GetNumberOfTuples();
                    scalars         = scipy.zeros((rows,cols));
                
                for j in xrange(len(scalarTuple)):
                    scalars[j,i]    = scalarTuple[j];

        self.scalarData             = scalars;

    def __LaplacianMatrix(self):
        numDims                     = self.polygonData.shape[0];
        numPoints                   = self.pointData.shape[1];
        numPolygons                 = self.polygonData.shape[1];
        boundary                    = self.boundary;
        boundaryConstrain           = scipy.zeros((2,numPoints));

        sparseMatrix                = scipy.sparse.csr_matrix((numPoints, numPoints));

        for i in range(0, numDims):
            i1                      = (i + 0)%3;
            i2                      = (i + 1)%3;
            i3                      = (i + 2)%3;

            distP2P1                = self.pointData[:, self.polygonData[i2, :]]                \
                                      - self.pointData[:, self.polygonData[i1, :]];
            distP3P1                = self.pointData[:, self.polygonData[i3, :]]                \
                                      - self.pointData[:, self.polygonData[i1, :]];

            distP2P1                = distP2P1 / numpy.matlib.repmat(scipy.sqrt((distP2P1**2).sum(0)), 3, 1);
            distP3P1                = distP3P1 / numpy.matlib.repmat(scipy.sqrt((distP3P1**2).sum(0)), 3, 1);

            angles                  = scipy.arccos((distP2P1 * distP3P1).sum(0));

            iterData1               = scipy.sparse.csr_matrix((1/scipy.tan(angles),         \
                                                              (self.polygonData[i2,:],      \
                                                              self.polygonData[i3,:])),     \
                                                              shape=(numPoints,            \
                                                                     numPoints));

            iterData2               = scipy.sparse.csr_matrix((1/scipy.tan(angles),     \
                                                              (self.polygonData[i3,:],  \
                                                              self.polygonData[i2,:])), \
                                                              shape=(numPoints,        \
                                                                     numPoints));

            sparseMatrix            = sparseMatrix + iterData1 + iterData2;

        diagonal                    = sparseMatrix.sum(0);
        diagonalSparse              = scipy.sparse.spdiags(diagonal, 0, \
                                                           numPoints,  \
                                                           numPoints);
        self.laplacianMatrix        = diagonalSparse - sparseMatrix;

    def __CalculateLinearTransformation(self):
        if self.laplacianMatrix is not None:
            if self.boundary is not None:
                laplacian                               = self.laplacianMatrix;
                laplacian[self.boundary, :]             = 0;
                laplacian[self.boundary, self.boundary] = 1;

                Z                                       = self.GetWithinBoundarySinCos();

                boundaryConstrain[:, boundary]          = Z;
                self.linearMatrix                       = boundaryConstrain;

                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa
                # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa

    def __CalculateBoundary(self):
        startingPoint                               = None;
        currentPoint                                = None;
        foundBoundary                               = False;
        cellId                                      = 0;
        boundary                                    = [];
        visitedEdges                                = [];
        visitedPoints                               = [];
        visitedBoundaryEdges                        = [];

        for cellId in xrange(self.originalPolyData.GetNumberOfCells()):
            cellPointIdList                         = vtk.vtkIdList();
            cellEdges                               = [];

            try:
                a = self.originalPolyData.GetCellPoints(cellId, cellPointIdList);
            except:
                raise Exception("Cell ID does not exist. Possibly mesh provided "  \
                                + "is empty.");

            try:
                cellEdges = [[cellPointIdList.GetId(0), cellPointIdList.GetId(1)], \
                             [cellPointIdList.GetId(1), cellPointIdList.GetId(2)], \
                             [cellPointIdList.GetId(2), cellPointIdList.GetId(0)]];
            except:
                raise Exception("Mesh does not contain a triangulation. "          \
                                + "Check input.");

            for i in xrange(len(cellEdges)):
                if (cellEdges[i] in visitedEdges)   == False:
                    visitedEdges.append(cellEdges[i]);

                    edgeIdList                      = vtk.vtkIdList()
                    a = edgeIdList.InsertNextId(cellEdges[i][0]);
                    a = edgeIdList.InsertNextId(cellEdges[i][1]);

                    singleCellEdgeNeighborIds       = vtk.vtkIdList();

                    a = self.originalPolyData.GetCellEdgeNeighbors(cellId,                 \
                                                      cellEdges[i][0],             \
                                                      cellEdges[i][1],             \
                                                      singleCellEdgeNeighborIds);

                    if singleCellEdgeNeighborIds.GetNumberOfIds() == 0:
                        foundBoundary               = True;

                        startingPoint               = cellEdges[i][0];
                        currentPoint                = cellEdges[i][1];

                        boundary.append(cellEdges[i][0]);
                        boundary.append(cellEdges[i][1]);

                        visitedBoundaryEdges.append([currentPoint,startingPoint]);
                        visitedBoundaryEdges.append([startingPoint,currentPoint]);

            if foundBoundary == True:
                break;

        if foundBoundary == False:
            raise Exception("The mesh provided has no boundary; not possible to "  \
                            + "do Quasi-Conformal Mapping on this dataset.");

        while currentPoint != startingPoint:
            neighboringCells                        = vtk.vtkIdList();

            self.originalPolyData.GetPointCells(currentPoint, neighboringCells);

            for i in xrange(neighboringCells.GetNumberOfIds()):
                cell                                = neighboringCells.GetId(i);
                triangle                            = self.originalPolyData.GetCell(cell);

                for j in xrange(triangle.GetNumberOfPoints()):
                    if triangle.GetPointId(j) == currentPoint:
                        j1                          = (j + 1) % 3;
                        j2                          = (j + 2) % 3;

                        edge1                       = [triangle.GetPointId(j),
                                                       triangle.GetPointId(j1)];
                        edge2                       = [triangle.GetPointId(j),
                                                       triangle.GetPointId(j2)];

                edgeNeighbors1                      = vtk.vtkIdList();
                edgeNeighbors2                      = vtk.vtkIdList();

                a = self.originalPolyData.GetCellEdgeNeighbors(cell, edge1[0], edge1[1],    \
                                                       edgeNeighbors1);

                a = self.originalPolyData.GetCellEdgeNeighbors(cell, edge2[0], edge2[1],    \
                                                       edgeNeighbors2);

                if edgeNeighbors1.GetNumberOfIds() == 0:
                    if ([edge1[1], edge1[0]] in visitedBoundaryEdges) == False:
                        if (edge1[1] in boundary) == False:
                            boundary.append(edge1[1]);
                        visitedBoundaryEdges.append([edge1[0], edge1[1]])
                        visitedBoundaryEdges.append([edge1[1], edge1[0]])
                        currentPoint                = edge1[1];
                        break;

                if edgeNeighbors2.GetNumberOfIds() == 0:
                    if ([edge2[1], edge2[0]] in visitedBoundaryEdges) == False:
                        if (edge2[1] in boundary) == False:
                            boundary.append(edge2[1]);
                        visitedBoundaryEdges.append([edge2[0], edge2[1]])
                        visitedBoundaryEdges.append([edge2[1], edge2[0]])
                        currentPoint                = edge2[1];
                        break;

        self.boundary                               = scipy.asarray(boundary, dtype=int);

    def GetPolyData(self):
        return self.originalPolyData;

    def GetPointData(self):
        return self.pointData;

    def GetImageType(self):
        return self.imageType;

    def GetPath(self):
        return self.path;

    def GetNormalData(self):
        return self.normalData;

    def GetPolygonData(self):
        return self.polygonData;

    def GetNumberOfPoints(self):
        return self.__nPoints;

    def GetNumberOfPolygons(self):
        return self.__nPolygons;

    def GetSeptumId(self):
        return self.septum;

    def GetLaplacianMatrix(self):
        return self.laplacianMatrix;

    def GetBoundary(self):
        return self.boundary;

    def GetBoundaryPoints(self):
        return self.pointData[:, self.boundary];

    def FlipBoundary(self):
        self.boundary       = scipy.flip(self.boundary, 0);
        self.boundary       = scipy.roll(self.boundary, 1);

    def GetWithinBoundaryDistances(self):
        boundaryNext                    = scipy.roll(self.boundary, -1);
        boundaryNextPoints              = self.pointData[:, boundaryNext];

        distanceToNext                  = boundaryNextPoints - self.GetBoundaryPoints();

        return scipy.sqrt((distanceToNext**2).sum(0));

    def GetPerimeter(self):
        return self.GetWithinBoundaryDistances().sum();

    def GetWithinBoundaryDistancesAsFraction(self):
        euclideanNorm                   = self.GetWithinBoundaryDistances();
        perimeter                       = euclideanNorm.sum();

        return euclideanNorm/perimeter;

    def GetWithinBoundaryAngles(self):
        circleLength                    = 2*scipy.pi;
        fraction                        = self.GetWithinBoundaryDistancesAsFraction();

        angles                          = scipy.cumsum(circleLength*fraction);
        angles                          = scipy.roll(angles, 1);
        angles[0]                       = 0;

        return angles;

    def GetWithinBoundarySinCos(self):
        angles                          = self.GetWithinBoundaryAngles();
        Z                               = scipy.zeros((2, angles.size));
        Z[0,:]                          = scipy.cos(angles);
        Z[1,:]                          = scipy.sin(angles);

        return Z;

    def RearrangeBoundary(self, objectivePoint=None):
        septalIndex                             = None;
        septalPoint                             = None;
        closestPoint                            = None;

        if objectivePoint is None:
            if self.septum is None:
                raise Exception("No septal point provided in function call "    \
                                "and no septal point provided in constructor. " \
                                "Aborting arrangement. ");
            else:
                septalIndex                     = self.septum;
        else:
            print("Using provided septal point as rearranging point.");
            self.septum                         = objectivePoint;
            septalIndex                         = objectivePoint;

        if septalIndex in self.boundary:
            closestPoint                        = septalIndex;
            closestPointIndex                   = scipy.where(self.boundary==septalIndex);

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex           = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one "    \
                           + "point ID associated to the objective point. Check your "      \
                           + "input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point "  \
                       + "ID associated to the objective point. Check your input data or "  \
                       + "contact the maintainer.");

            self.boundary                       = scipy.roll(self.boundary, 
                                                             -closestPointIndex);
        else:
            try:
                septalPoint                     = self.pointData[:, septalIndex];
            except:
                raise Exception("Septal point provided out of data bounds; the "   \
                       + "point does not exist (it is out of bounds) or a point "  \
                       + "identifier beyond the total amount of points has been "  \
                       + "provided. Check input.");

            if len(self.boundary.shape) == 1:
                septalPoint                     = numpy.matlib.repmat(septalPoint,
                                                              self.boundary.size, 1);
                septalPoint                     = septalPoint .transpose();
            else:
                raise Exception("It seems you have multiple boundaries. "          \
                                + "Contact the package maintainer.");

            distanceToObjectivePoint            = (self.pointData[:, self.boundary] - septalPoint);
            distanceToObjectivePoint            = scipy.sqrt((distanceToObjectivePoint**2).sum(0));
            closestPointIndex                   = scipy.where(distanceToObjectivePoint          \
                                                      == distanceToObjectivePoint.min());
            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex           = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one "    \
                           + "point ID associated to the objective point. Check your "      \
                           + "input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point "  \
                       + "ID associated to the objective point. Check your input data or "  \
                       + "contact the maintainer.");

            self.boundary                       = scipy.roll(self.boundary, 
                                                             -closestPointIndex);

    # def flattening(self):






print(time.time() - start);



septum                      = 201479 - 1;

path                        = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
start = time.time(); MRI1   = VentricularImage(path, septum); print(time.time() - start);
start = time.time(); MRI2   = VentricularImage(path); print(time.time() - start);














































































# from __future__ import division

# import os;
# import numpy;
# import scipy;

# from numpy.matlib import repmat;
# from numpy import int;
# from scipy import roll;
# from scipy import flip;
# from scipy import zeros;
# from scipy import arccos;
# from scipy import sin;
# from scipy import cos;
# from scipy import sqrt;
# from scipy import asarray;
# from scipy import tan;
# from scipy import pi;
# from scipy import cumsum;
# from scipy import where;

# from scipy.sparse import csr_matrix;
# from scipy.sparse import spdiags;
# from scipy.sparse.linalg import spsolve; 

# import time;
# import vtk;


# start = time.time();

# class VentricularImage:
#     """ DOCSTRING """

#     imageType                       = None;

#     path                            = None;
#     originalPolyData                = None;
#     QCMPolyData                     = None;

#     pointData                       = None;
#     polygonData                     = None;

#     scalarData                      = None;
#     normalData                      = None;

#     __nPoints                       = None;
#     __nPolygons                     = None;
#     __nDims                         = None;

#     septum                          = None;

#     laplacianMatrix                 = None;
#     linearMatrix                    = None;

#     boundary                        = None;
#     # boundaryPoints                  = None;

#     def __init__(self, path=None, septum=None):
#         if path is None:
#             if septum is not None:
#                 self.septum         = septum;
#             # if septum is None:
#             #     self.imageType          = None;

#             #     self.path               = None;
#             #     self.originalPolyData   = None;
#             #     self.QCMPolyData        = None;

#             #     self.pointData          = None;
#             #     self.polygonData        = None;

#             #     self.__nPoints          = None;
#             #     self.__nPolygons        = None;
#             #     self.__nDims            = None;

#             #     self.normalData         = None;
#             #     self.scalarData         = None;

#             #     self.septum             = None;

#             #     self.laplacianMatrix    = None;
#             #     self.linearMatrix       = None;

#             #     self.boundary           = None;
#             #     # self.boundaryPoints     = None;
#             # else:
#             #     self.imageType          = None;

#             #     self.path               = None;
#             #     self.originalPolyData   = None;
#             #     self.QCMPolyData        = None;

#             #     self.pointData          = None;
#             #     self.polygonData        = None;

#             #     self.__nPoints          = None;
#             #     self.__nPolygons        = None;
#             #     self.__nDims            = None;

#             #     self.normalData         = None;
#             #     self.scalarData         = None;

#             #     self.septum             = septum;

#             #     self.laplacianMatrix    = None;
#             #     self.linearMatrix       = None;

#             #     self.boundary           = None;
#             #     # self.boundaryPoints     = None;

#         else:
#             if os.path.isfile(path):
#                 self.path           = path;
#             else:
#                 raise RuntimeError("File does not exist. If file exists and this"  \
#                                    + "error was raised when automatically "        \
#                                    + "scanning documents in a folder, contact "    \
#                                    + "the package maintainer.");

#             self.__ReadPolyData();
#             self.__ReadPointData();
#             self.__ReadPolygonData();
#             self.__ReadNormalData();
#             self.__ReadScalarData();
#             self.septum             = septum;
#             self.__nPoints          = self.pointData.shape[1];
#             self.__nPolygons        = self.polygonData.shape[1];
#             self.__CalculateBoundary();
#             self.__LaplacianMatrix();

#             # if septum is None:
#             #     self.__ReadPolyData();
#             #     self.__ReadPointData();
#             #     self.__ReadPolygonData();
#             #     self.__ReadNormalData();
#             #     self.__ReadScalarData();
#             #     self.septum             = None;
#             #     self.__nPoints          = self.pointData.shape[1];
#             #     self.__nPolygons        = self.polygonData.shape[1];
#             #     self.__CalculateBoundary();
#             #     self.__LaplacianMatrix();
#             # else:
#             #     self.__ReadPolyData();
#             #     self.__ReadPointData();
#             #     self.__ReadPolygonData();
#             #     self.__ReadNormalData();
#             #     self.__ReadScalarData();
#             #     self.septum             = septum;
#             #     self.__nPoints          = self.pointData.shape[1];
#             #     self.__nPolygons        = self.polygonData.shape[1];
#             #     self.__CalculateBoundary();
#             #     self.__LaplacianMatrix();

#     def __ReadPolyData(self):
#         reader                      = vtk.vtkPolyDataReader();
#         reader.SetFileName(self.path);
#         reader.Update();

#         polyData                    = reader.GetOutput();
#         polyData.BuildLinks();

#         self.originalPolyData       = polyData;

#     def __ReadPolygonData(self):
#         rows                        = None;
#         cols                        = None;
#         polygons                    = None;

#         try:
#             polys = self.originalPolyData.GetPolys();
#         except:
#             raise RuntimeError("Tried to call function 'extract_polygons' with a " \
#                                + "variable with no 'GetPolys()' method.");

#         for i in xrange(self.originalPolyData.GetNumberOfCells()):
#             triangle                = self.originalPolyData.GetCell(i);
#             pointIds                = triangle.GetPointIds();

#             if polygons is None:
#                 rows                = pointIds.GetNumberOfIds();
#                 cols                = self.originalPolyData.GetNumberOfCells();
#                 polygons            = scipy.zeros((rows,cols), dtype=numpy.int);
            
#             polygons[0,i]           = pointIds.GetId(0);
#             polygons[1,i]           = pointIds.GetId(1);
#             polygons[2,i]           = pointIds.GetId(2);

#         self.polygonData            = polygons;

#     def __ReadPointData(self):
#         rows                        = None;
#         cols                        = None;
#         points                      = None;

#         try:
#             pointVector             = self.originalPolyData.GetPoints();
#         except:
#             raise RuntimeError("Tried to call function 'extract_points' with a "   \
#                                + "variable with no 'GetPoints()' method.");

#         if pointVector:
#             for i in range(0, pointVector.GetNumberOfPoints()):
#                 point_tuple         = pointVector.GetPoint(i);

#                 if points is None:
#                     rows            = len(point_tuple);
#                     cols            = pointVector.GetNumberOfPoints();
#                     points          = scipy.zeros((rows,cols));
                
#                 points[0,i]         = point_tuple[0];
#                 points[1,i]         = point_tuple[1];
#                 points[2,i]         = point_tuple[2];

#         self.pointData              = points;

#     def __ReadNormalData(self):
#         rows                        = None;
#         cols                        = None;
#         normals                     = None;

#         try:
#             normalVector            = self.originalPolyData.GetPointData().GetNormals();
#         except:
#             raise RuntimeError("Tried to call function 'extract_normals' with a "  \
#                                + "variable with no 'GetNormals()' method.");

#         if normalVector:
#             for i in range(0, normalVector.GetNumberOfTuples()):
#                 normalTuple         = normalVector.GetTuple(i);

#                 if normals is None:
#                     rows            = len(normalTuple);
#                     cols            = normalVector.GetNumberOfTuples();
#                     normals         = scipy.zeros((rows,cols));
                
#                 normals[0,i]        = normalTuple[0];
#                 normals[1,i]        = normalTuple[1];
#                 normals[2,i]        = normalTuple[2];

#         self.normalData             = normals;

#     def __ReadScalarData(self):
#         rows                        = None;
#         cols                        = None;
#         scalars                     = None;

#         try:
#             scalarVector            = self.originalPolyData.GetPointData().GetScalars();
#         except:
#             raise RuntimeError("Tried to call function 'extract_normals' with a "  \
#                                + "variable with no 'GetScalars()' method.");

#         if scalarVector:
#             for i in xrange(scalarVector.GetNumberOfTuples()):
#                 scalarTuple         = scalarVector.GetTuple(i);

#                 if scalars is None:
#                     rows            = len(scalarTuple);
#                     cols            = scalarVector.GetNumberOfTuples();
#                     scalars         = scipy.zeros((rows,cols));
                
#                 for j in xrange(len(scalarTuple)):
#                     scalars[j,i]    = scalarTuple[j];

#         self.scalarData             = scalars;

#     def __LaplacianMatrix(self):
#         numDims                     = self.polygonData.shape[0];
#         numPoints                   = self.pointData.shape[1];
#         numPolygons                 = self.polygonData.shape[1];
#         boundary                    = self.boundary;
#         boundaryConstrain           = scipy.zeros((2,numPoints));

#         sparseMatrix                = scipy.sparse.csr_matrix((numPoints, numPoints));

#         for i in range(0, numDims):
#             i1                      = (i + 0)%3;
#             i2                      = (i + 1)%3;
#             i3                      = (i + 2)%3;

#             distP2P1                = self.pointData[:, self.polygonData[i2, :]]                \
#                                       - self.pointData[:, self.polygonData[i1, :]];
#             distP3P1                = self.pointData[:, self.polygonData[i3, :]]                \
#                                       - self.pointData[:, self.polygonData[i1, :]];

#             distP2P1                = distP2P1 / numpy.matlib.repmat(scipy.sqrt((distP2P1**2).sum(0)), 3, 1);
#             distP3P1                = distP3P1 / numpy.matlib.repmat(scipy.sqrt((distP3P1**2).sum(0)), 3, 1);

#             angles                  = scipy.arccos((distP2P1 * distP3P1).sum(0));

#             iterData1               = scipy.sparse.csr_matrix((1/scipy.tan(angles),         \
#                                                               (self.polygonData[i2,:],      \
#                                                               self.polygonData[i3,:])),     \
#                                                               shape=(numPoints,            \
#                                                                      numPoints));

#             iterData2               = scipy.sparse.csr_matrix((1/scipy.tan(angles),     \
#                                                               (self.polygonData[i3,:],  \
#                                                               self.polygonData[i2,:])), \
#                                                               shape=(numPoints,        \
#                                                                      numPoints));

#             sparseMatrix            = sparseMatrix + iterData1 + iterData2;

#         diagonal                    = sparseMatrix.sum(0);
#         diagonalSparse              = scipy.sparse.spdiags(diagonal, 0, \
#                                                            numPoints,  \
#                                                            numPoints);
#         self.laplacianMatrix        = diagonalSparse - sparseMatrix;

#     def __CalculateLinearTransformation(self):
#         if self.laplacianMatrix is not None:
#             if self.boundary is not None:
#                 laplacian                               = self.laplacianMatrix;
#                 laplacian[self.boundary, :]             = 0;
#                 laplacian[self.boundary, self.boundary] = 1;

#                 Z                                       = self.GetWithinBoundarySinCos();

#                 boundaryConstrain[:, boundary]          = Z;
#                 self.linearMatrix                       = boundaryConstrain;

#     def __CalculateBoundary(self):
#         startingPoint                               = None;
#         currentPoint                                = None;
#         foundBoundary                               = False;
#         cellId                                      = 0;
#         boundary                                    = [];
#         visitedEdges                                = [];
#         visitedPoints                               = [];
#         visitedBoundaryEdges                        = [];

#         for cellId in xrange(self.originalPolyData.GetNumberOfCells()):
#             cellPointIdList                         = vtk.vtkIdList();
#             cellEdges                               = [];

#             try:
#                 a = self.originalPolyData.GetCellPoints(cellId, cellPointIdList);
#             except:
#                 raise Exception("Cell ID does not exist. Possibly mesh provided "  \
#                                 + "is empty.");

#             try:
#                 cellEdges = [[cellPointIdList.GetId(0), cellPointIdList.GetId(1)], \
#                              [cellPointIdList.GetId(1), cellPointIdList.GetId(2)], \
#                              [cellPointIdList.GetId(2), cellPointIdList.GetId(0)]];
#             except:
#                 raise Exception("Mesh does not contain a triangulation. "          \
#                                 + "Check input.");

#             for i in xrange(len(cellEdges)):
#                 if (cellEdges[i] in visitedEdges)   == False:
#                     visitedEdges.append(cellEdges[i]);

#                     edgeIdList                      = vtk.vtkIdList()
#                     a = edgeIdList.InsertNextId(cellEdges[i][0]);
#                     a = edgeIdList.InsertNextId(cellEdges[i][1]);

#                     singleCellEdgeNeighborIds       = vtk.vtkIdList();

#                     a = self.originalPolyData.GetCellEdgeNeighbors(cellId,                 \
#                                                       cellEdges[i][0],             \
#                                                       cellEdges[i][1],             \
#                                                       singleCellEdgeNeighborIds);

#                     if singleCellEdgeNeighborIds.GetNumberOfIds() == 0:
#                         foundBoundary               = True;

#                         startingPoint               = cellEdges[i][0];
#                         currentPoint                = cellEdges[i][1];

#                         boundary.append(cellEdges[i][0]);
#                         boundary.append(cellEdges[i][1]);

#                         visitedBoundaryEdges.append([currentPoint,startingPoint]);
#                         visitedBoundaryEdges.append([startingPoint,currentPoint]);

#             if foundBoundary == True:
#                 break;

#         if foundBoundary == False:
#             raise Exception("The mesh provided has no boundary; not possible to "  \
#                             + "do Quasi-Conformal Mapping on this dataset.");

#         while currentPoint != startingPoint:
#             neighboringCells                        = vtk.vtkIdList();

#             self.originalPolyData.GetPointCells(currentPoint, neighboringCells);

#             for i in xrange(neighboringCells.GetNumberOfIds()):
#                 cell                                = neighboringCells.GetId(i);
#                 triangle                            = self.originalPolyData.GetCell(cell);

#                 for j in xrange(triangle.GetNumberOfPoints()):
#                     if triangle.GetPointId(j) == currentPoint:
#                         j1                          = (j + 1) % 3;
#                         j2                          = (j + 2) % 3;

#                         edge1                       = [triangle.GetPointId(j),
#                                                        triangle.GetPointId(j1)];
#                         edge2                       = [triangle.GetPointId(j),
#                                                        triangle.GetPointId(j2)];

#                 edgeNeighbors1                      = vtk.vtkIdList();
#                 edgeNeighbors2                      = vtk.vtkIdList();

#                 a = self.originalPolyData.GetCellEdgeNeighbors(cell, edge1[0], edge1[1],    \
#                                                        edgeNeighbors1);

#                 a = self.originalPolyData.GetCellEdgeNeighbors(cell, edge2[0], edge2[1],    \
#                                                        edgeNeighbors2);

#                 if edgeNeighbors1.GetNumberOfIds() == 0:
#                     if ([edge1[1], edge1[0]] in visitedBoundaryEdges) == False:
#                         if (edge1[1] in boundary) == False:
#                             boundary.append(edge1[1]);
#                         visitedBoundaryEdges.append([edge1[0], edge1[1]])
#                         visitedBoundaryEdges.append([edge1[1], edge1[0]])
#                         currentPoint                = edge1[1];
#                         break;

#                 if edgeNeighbors2.GetNumberOfIds() == 0:
#                     if ([edge2[1], edge2[0]] in visitedBoundaryEdges) == False:
#                         if (edge2[1] in boundary) == False:
#                             boundary.append(edge2[1]);
#                         visitedBoundaryEdges.append([edge2[0], edge2[1]])
#                         visitedBoundaryEdges.append([edge2[1], edge2[0]])
#                         currentPoint                = edge2[1];
#                         break;

#         if self.septum is None:
#             self.boundary                           = scipy.asarray(boundary, dtype=int);
#         else:
#             self.boundary                           = scipy.asarray(boundary, dtype=int);
#             self.boundary                           = self.RearrangeBoundary();

#     def GetPolyData(self):
#         return self.originalPolyData;

#     def GetPointData(self):
#         return self.pointData;

#     def GetImageType(self):
#         return self.imageType;

#     def GetPath(self):
#         return self.path;

#     def GetNormalData(self):
#         return self.normalData;

#     def GetPolygonData(self):
#         return self.polygonData;

#     def GetNumberOfPoints(self):
#         return self.__nPoints;

#     def GetNumberOfPolygons(self):
#         return self.__nPolygons;

#     def GetSeptumId(self):
#         return self.septum;

#     def GetLaplacianMatrix(self):
#         return self.laplacianMatrix;

#     def GetBoundary(self):
#         return self.boundary;

#     def GetBoundaryPoints(self):
#         return self.pointData[:, self.boundary];

#     # def ReadPolyData(path):
#     #     self.__init__(path);

#     def FlipBoundary(self):
#         self.boundary       = scipy.flip(self.boundary, 0);
#         self.boundary       = scipy.roll(self.boundary, 1);
#         # self.boundaryPoints = self.pointData[:, self.boundary];

#     # def ReadPolyData(path):
#     #     reader = vtk.vtkPolyDataReader();
#     #     reader.SetFileName(path);
#     #     reader.Update();

#     #     self.polyData = reader.GetOutput();
#     #     self.polyData.BuildLinks();

#     # def SetPolyData(newPolyData):
#     #     self.imageType                  = None;
#     #     self.path                       = None;
#     #     self.originalPolyData           = newPolyData;
#     #     self.pointData                  = self.__ReadPointData();
#     #     self.polygonData                = self.__ReadPolygonData();
#     #     self.normalData                 = self.__ReadNormalData();
#     #     self.laplacianMatrix            = self.__LaplacianMatrix();
#     #     self.boundary                   = self.__CalculateBoundary();
#     #     self.boundaryPoints             = self.pointData[:, self.boundary];

#     def GetPerimeter(self):
#         boundaryNext                    = scipy.roll(self.boundary,-1);
#         boundaryNextPoints              = self.pointData[:, boundaryNext];

#         distanceToNext                  = boundaryNextPoints - self.GetBoundaryPoints();
#         euclideanNorm                   = scipy.sqrt((distanceToNext**2).sum(0));

#         return euclideanNorm.sum();

#     def GetWithinBoundaryDistances(self):
#         boundaryNext                    = scipy.roll(self.boundary,-1);
#         boundaryNextPoints              = self.pointData[:, boundaryNext];

#         distanceToNext                  = boundaryNextPoints - self.GetBoundaryPoints();

#         return scipy.sqrt((distanceToNext**2).sum(0));

#     def GetWithinBoundaryDistancesAsFraction(self):
#         boundaryNext                    = scipy.roll(self.boundary,-1);
#         boundaryNextPoints              = self.pointData[:, boundaryNext];

#         distanceToNext                  = boundaryNextPoints - self.GetBoundaryPoints();

#         euclideanNorm                   = scipy.sqrt((distanceToNext**2).sum(0));
#         perimeter                       = euclideanNorm.sum();

#         return euclideanNorm/perimeter;

#     def GetWithinBoundaryAngles(self):
#         circleLength                    = scipy.pi;
#         fraction                        = self.GetWithinBoundaryDistancesAsFraction();

#         angles                          = scipy.cumsum(circleLength*fraction);
#         angles                          = scipy.roll(angles, 1);
#         angles[0]                       = 0;

#         return angles;

#     def GetWithinBoundarySinCos(self):
#         angles                          = self.GetWithinBoundaryAngles();
#         Z                               = scipy.zeros((2, angles.size));
#         Z[0,:]                          = scipy.cos(angles);
#         Z[1,:]                          = scipy.sin(angles);

#         return Z;

#     def RearrangeBoundary(self, objectivePoint=None):
#         septalIndex                             = None;
#         septalPoint                             = None;
#         closestPoint                            = None;

#         if objectivePoint is None:
#             if self.septum is None:
#                 raise Exception("No septal point provided in function call " \
#                                 "and no septal point provided in constructor.");
#             else:
#                 septalIndex                     = self.septum;
#         else:
#             print("Using provided septal point as rearranging point.");
#             self.septum                         = objectivePoint;
#             septalIndex                         = objectivePoint;

#         # try:
#         #     septalPoint                         = self.pointData[:, objectivePoint];
#         # except:
#         #     raise Exception("Septal point provided out of data bounds; the point " \
#         #            + "does not exist (it is out of bounds) or a point identifier " \
#         #            + "beyond the total amount of points has been provided. Check " \
#         #            + "input.");

#         if septalIndex in self.boundary:
#             closestPoint                        = septalIndex;
#             closestPointIndex                   = scipy.where(self.boundary==septalIndex);

#             if len(closestPointIndex) == 1:
#                 if len(closestPointIndex[0]) == 1:
#                     closestPointIndex           = closestPointIndex[0][0];
#                 else:
#                     raise Exception("It seems your vtk file has more than one "    \
#                            + "point ID associated to the objective point. Check your "      \
#                            + "input data or contact the maintainer.");
#             else:
#                 raise Exception("It seems your vtk file has more than one point "  \
#                        + "ID associated to the objective point. Check your input data or "  \
#                        + "contact the maintainer.");

#             self.boundary                       = scipy.roll(self.boundary, 
#                                                              -closestPointIndex);
#             # self.boundaryPoints                 = self.pointData[:, self.boundary];
#         else:
#             try:
#                 septalPoint                     = self.pointData[:, septalIndex];
#             except:
#                 raise Exception("Septal point provided out of data bounds; the "   \
#                        + "point does not exist (it is out of bounds) or a point "  \
#                        + "identifier beyond the total amount of points has been "  \
#                        + "provided. Check input.");
#             # septalPoint                 = self.pointData[:, septalIndex];

#             if len(self.boundary.shape) == 1:
#                 septalPoint                     = numpy.matlib.repmat(septalPoint,
#                                                               self.boundary.size, 1);
#                 septalPoint                     = septalPoint .transpose();
#             else:
#                 raise Exception("It seems you have multiple boundaries. "          \
#                                 + "Contact the package maintainer.");

#             distanceToObjectivePoint            = (self.pointData[:, self.boundary] - septalPoint);
#             distanceToObjectivePoint            = scipy.sqrt((distanceToObjectivePoint**2).sum(0));
#             closestPointIndex                   = scipy.where(distanceToObjectivePoint          \
#                                                       == distanceToObjectivePoint.min());
#             if len(closestPointIndex) == 1:
#                 if len(closestPointIndex[0]) == 1:
#                     closestPointIndex           = closestPointIndex[0][0];
#                 else:
#                     raise Exception("It seems your vtk file has more than one "    \
#                            + "point ID associated to the objective point. Check your "      \
#                            + "input data or contact the maintainer.");
#             else:
#                 raise Exception("It seems your vtk file has more than one point "  \
#                        + "ID associated to the objective point. Check your input data or "  \
#                        + "contact the maintainer.");

#             self.boundary                       = scipy.roll(self.boundary, 
#                                                              -closestPointIndex);
#             # self.boundaryPoints                 = self.pointData[:, self.boundary];

#     # def flattening(self):






# print(time.time() - start);



# septum                      = 201479 - 1;

# path                        = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
# start = time.time(); MRI1   = VentricularImage(path, septum); print(time.time() - start);
# start = time.time(); MRI2   = VentricularImage(path); print(time.time() - start);




































































# # start                       = time.time();
# # # path                        = os.path.join("/home/bee/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
# # septum                      = 201479 - 1;


# # path                        = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

# # polyData                    = reader_vtk(path);
# # point_data                  = extract_points(polyData);
# # polygon_data                = extract_polygons(polyData);
# # normal_data                 = extract_normals(polyData);

# # boundary                    = compute_boundary(polyData);
# # boundary_points             = point_data[:, boundary];

# # boundary                    = flippity_flop(boundary);

# # boundary                    = rearrange_boundary(boundary, point_data, septum);
# # boundary_points             = point_data[:, boundary];

# # boundary                    = rearrange_boundary(boundary, point_data, septum);
# # boundary_points             = point_data[:, boundary];


# # L                           = laplacian_matrix(polyData);
# # find_L                      = scipy.sparse.find(L);
# # print(time.time() - start);

















