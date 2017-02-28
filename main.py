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
import collections;

from numpy.matlib import repmat;
from scipy.sparse import coo_matrix;
# from scipy.sparse import csr_matrix;
# from scipy.sparse import csc_matrix;
# from scipy.sparse import lil_matrix;
# from scipy.sparse import csgraph;
# from scipy.special import cotdg;

import time;
import vtk;


class VentricularImage:
    """ DOCSTRING """

    imageType                   = None;

    path                        = None;
    polyData                    = None;

    pointData                   = None;
    polygonData                 = None;
    normalData                  = None;

    nPoints                     = None;
    nPolygons                   = None;

    laplacianMatrix             = None;

    boundary                    = None;
    boundaryPoints              = None;

    def __init__(self):
        self.imageType          = None;

        self.path               = None;
        self.polyData           = None;

        self.pointData          = None;
        self.polygonData        = None;
        self.normalData         = None;

        self.laplacianMatrix    = None;

        self.boundary           = None;
        self.boundaryPoints     = None;

    def __init__(self, path=None):
        if path is None:
            self.imageType          = None;

            self.path               = None;
            self.polyData           = None;

            self.pointData          = None;
            self.polygonData        = None;
            self.normalData         = None;

            self.laplacianMatrix    = None;

            self.boundary           = None;
            self.boundaryPoints     = None;
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
            self.__LaplacianMatrix();
            self.__CalculateBoundary();
            self.boundaryPoints     = self.pointData[:, self.boundary];

    def __ReadPolyData(self):
        reader                      = vtk.vtkPolyDataReader();
        reader.SetFileName(self.path);
        reader.Update();

        polyData                    = reader.GetOutput();
        polyData.BuildLinks();

        self.polyData               = polyData;

    def __ReadPolygonData(self):
        rows                = None;
        cols                = None;
        polygons            = None;

        try:
            polys = self.polyData.GetPolys();
        except:
            raise RuntimeError("Tried to call function 'extract_polygons' with a " \
                               + "variable with no 'GetPolys()' method.");

        for i in xrange(self.polyData.GetNumberOfCells()):
            triangle            = self.polyData.GetCell(i);
            pointIds            = triangle.GetPointIds();

            if polygons is None:
                rows            = pointIds.GetNumberOfIds();
                cols            = self.polyData.GetNumberOfCells();
                polygons        = scipy.zeros((rows,cols), dtype=numpy.int);
            
            polygons[0,i]       = pointIds.GetId(0);
            polygons[1,i]       = pointIds.GetId(1);
            polygons[2,i]       = pointIds.GetId(2);

        self.polygonData        = polygons;

    def __ReadPointData(self):
        rows                        = None;
        cols                        = None;
        points                      = None;

        try:
            pointVector             = (self.polyData).GetPoints();
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
            normalVector            = self.polyData.GetPointData().GetNormals();
        except:
            raise RuntimeError("Tried to call function 'extract_normals' with a "  \
                               + "variable with no 'GetNormals()' method.");

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

        self.normalData             = normals;

    def __LaplacianMatrix(self):
        num_dims                = self.polygonData.shape[0];
        num_points              = self.pointData.shape[1];
        num_polygons            = self.polygonData.shape[1];

        sparse_matrix           = scipy.sparse.coo_matrix((num_points, num_points));

        for i in range(0, num_dims):
            i1                  = (i + 0)%3;
            i2                  = (i + 1)%3;
            i3                  = (i + 2)%3;

            dist_p2_p1          = self.pointData[:, self.polygonData[i2, :]]                \
                                  - self.pointData[:, self.polygonData[i1, :]];
            dist_p3_p1          = self.pointData[:, self.polygonData[i3, :]]                \
                                  - self.pointData[:, self.polygonData[i1, :]];

            dist_p2_p1          = dist_p2_p1 / numpy.matlib.repmat(scipy.sqrt((dist_p2_p1**2).sum(0)), 3, 1);
            dist_p3_p1          = dist_p3_p1 / numpy.matlib.repmat(scipy.sqrt((dist_p3_p1**2).sum(0)), 3, 1);

            angles              = scipy.arccos((dist_p2_p1 * dist_p3_p1).sum(0));

            iter_data1          = scipy.sparse.coo_matrix((1/scipy.tan(angles),     \
                                                          (self.polygonData[i2,:],      \
                                                          self.polygonData[i3,:])),     \
                                                          shape=(num_points,        \
                                                                 num_points));

            iter_data2          = scipy.sparse.coo_matrix((1/scipy.tan(angles), \
                                                          (self.polygonData[i3,:],  \
                                                          self.polygonData[i2,:])), \
                                                          shape=(num_points,    \
                                                                 num_points));

            sparse_matrix       = sparse_matrix + iter_data1 + iter_data2;

        diagonal                = sparse_matrix.sum(0);
        diagonal_sparse         = scipy.sparse.spdiags(diagonal,0,num_points,num_points);

        self.laplacianMatrix    = diagonal_sparse - sparse_matrix;

    def __CalculateBoundary(self):
        startingPoint                               = None;
        currentPoint                                = None;
        foundBoundary                               = False;
        cellId                                      = 0;
        boundary                                    = [];
        visitedEdges                                = [];
        visitedPoints                               = [];
        visitedBoundaryEdges                        = [];

        for cellId in xrange(self.polyData.GetNumberOfCells()):
            cellPointIdList                         = vtk.vtkIdList();
            cellEdges                               = [];

            try:
                a = self.polyData.GetCellPoints(cellId, cellPointIdList);
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

                    a = self.polyData.GetCellEdgeNeighbors(cellId,                 \
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

            self.polyData.GetPointCells(currentPoint, neighboringCells);

            for i in xrange(neighboringCells.GetNumberOfIds()):
                cell                                = neighboringCells.GetId(i);
                triangle                            = self.polyData.GetCell(cell);

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

                a = self.polyData.GetCellEdgeNeighbors(cell, edge1[0], edge1[1],    \
                                                  edgeNeighbors1);

                a = self.polyData.GetCellEdgeNeighbors(cell, edge2[0], edge2[1],    \
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
        return self.polyData;

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

    def GetLaplacianMatrix(self):
        return self.laplacianMatrix;

    def GetBoundary(self):
        return self.boundary;

    def GetBoundaryPoints(self):
        return self.boundaryPoints;

    def ReadPolyData(path):
        self.__init__(path);

    def FlipBoundary(self):
        self.boundary       = numpy.flip(boundary, 0);
        self.boundaryPoints = self.pointData[:, self.boundary];

    # def ReadPolyData(path):
    #     reader = vtk.vtkPolyDataReader();
    #     reader.SetFileName(path);
    #     reader.Update();

    #     self.polyData = reader.GetOutput();
    #     self.polyData.BuildLinks();

    def SetPolyData(polyData):
        self.imageType          = None;
        self.path               = None;
        self.polyData           = polyData;
        self.pointData          = self.__ReadPointData();
        self.polygonData        = self.__ReadPolygonData();
        self.normalData         = self.__ReadNormalData();
        self.laplacianMatrix    = self.__LaplacianMatrix();
        self.boundary           = self.__CalculateBoundary();
        self.boundaryPoints     = self.pointData[:, self.boundary];

    def GetPerimeter(self):
        boundaryNext                    = numpy.roll(self.boundary,-1);
        boundaryNextPoints              = self.pointData[:, boundaryNext];

        distanceToNext                  = boundaryNextPoints - self.boundaryPoints;
        euclideanNorm                   = numpy.sqrt((distanceToNext**2).sum(0));

        return euclideanNorm.sum();

    def GetWithinBoundaryDistances(self):
        boundaryNext                    = numpy.roll(self.boundary,-1);
        boundaryNextPoints              = self.pointData[:, boundaryNext];

        distanceToNext                  = boundaryNextPoints - self.boundaryPoints;

        return numpy.sqrt((distanceToNext**2).sum(0));

    def GetWithinBoundaryDistancesAsPercentage(self):
        boundaryNext                    = numpy.roll(self.boundary,-1);
        boundaryNextPoints              = self.pointData[:, boundaryNext];

        distanceToNext                  = boundaryNextPoints - self.boundaryPoints;

        euclideanNorm                   = numpy.sqrt((distanceToNext**2).sum(0));
        perimeter                       = euclideanNorm.sum();

        return euclideanNorm/perimeter;

    def RearrangeBoundary(self, objectivePoint):
        septal_point                            = None;
        closest_point                           = None;

        try:
            septal_point                        = self.pointData[:, objectivePoint];
        except:
            raise Exception("Septal point provided out of data bounds; the point " \
                   + "does not exist (it is out of bounds) or a point identifier " \
                   + "beyond the total amount of points has been provided. Check " \
                   + "input.");

        if objectivePoint in self.boundary:
            closest_point                       = objectivePoint;
            closest_point_index                 = scipy.where(self.boundary==objectivePoint);

            if len(closest_point_index) == 1:
                if len(closest_point_index[0]) == 1:
                    closest_point_index         = closest_point_index[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one "    \
                           + "point ID associated to the objective point. Check your "      \
                           + "input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point "  \
                       + "ID associated to the objective point. Check your input data or "  \
                       + "contact the maintainer.");

            return numpy.roll(self.boundary, -closest_point_index);
        else:
            try:
                septal_point                    = self.pointData[:, objectivePoint];
            except:
                raise Exception("Septal point provided out of data bounds; the "   \
                       + "point does not exist (it is out of bounds) or a point "  \
                       + "identifier beyond the total amount of points has been "  \
                       + "provided. Check input.");
            # septal_point                = self.pointData[:, objectivePoint];

            if len(self.boundary.shape) == 1:
                septal_point                    = numpy.matlib.repmat(septal_point,
                                                              self.boundary.size, 1);
                septal_point                    = septal_point.transpose();
            else:
                raise Exception("It seems you have multiple boundaries. "          \
                                + "Contact the package maintainer.");

            distance_to_objectivePoint          = (self.pointData[:, self.boundary] - septal_point);
            distance_to_objectivePoint          = numpy.sqrt((distance_to_objectivePoint**2).sum(0));
            closest_point_index                 = scipy.where(distance_to_objectivePoint          \
                                                      == distance_to_objectivePoint.min());
            if len(closest_point_index) == 1:
                if len(closest_point_index[0]) == 1:
                    closest_point_index         = closest_point_index[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one "    \
                           + "point ID associated to the objective point. Check your "      \
                           + "input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point "  \
                       + "ID associated to the objective point. Check your input data or "  \
                       + "contact the maintainer.");

            self.boundary                       = numpy.roll(self.boundary, -closest_point_index);
            self.boundaryPoints                 = self.pointData[:, self.boundary];




print(time.time() - start);




path                        = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
start = time.time(); MRI = VentricularImage(path); print(time.time() - start);
































start                       = time.time();
# path                        = os.path.join("/home/bee/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
septum                      = 201479 - 1;


path                        = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

polyData                    = reader_vtk(path);
point_data                  = extract_points(polyData);
polygon_data                = extract_polygons(polyData);
normal_data                 = extract_normals(polyData);

boundary                    = compute_boundary(polyData);
boundary_points             = point_data[:, boundary];

boundary                    = flippity_flop(boundary);

boundary                    = rearrange_boundary(boundary, point_data, septum);
boundary_points             = point_data[:, boundary];

boundary                    = rearrange_boundary(boundary, point_data, septum);
boundary_points             = point_data[:, boundary];


L                           = laplacian_matrix(polyData);
find_L                      = scipy.sparse.find(L);
print(time.time() - start);

















