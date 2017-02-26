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



def reader_vtk(path):
    reader = vtk.vtkPolyDataReader();
    reader.SetFileName(path);
    reader.Update();

    polyData = reader.GetOutput();
    polyData.BuildLinks();

    return polyData;



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





# def compute_boundary(polyData):
#     boundary_points             = vtk.vtkIdList();

#     start = time.time();
#     for cellId in xrange(polyData.GetNumberOfCells()):
#         cellPointIds            = vtk.vtkIdList();

#         try:
#             triangle            = polyData.GetCell(cellId);
#             # polyData.GetCellPoints(cellId, cellPointIds);
#         except:
#             print("Tried to access a cell ID that is non-existant. Re-code \
#                    compute_boundary function. It seems that cell IDs are not \
#                    always consecutive as initially thought.");

#         edge1                   = triangle.GetEdge(0);
#         edge2                   = triangle.GetEdge(1);
#         edge3                   = triangle.GetEdge(2);

#     print(time.time() - start);



        # if cellPointIds.GetNumberOfIds() != 3:
        #     print("Non-consistent mesh shape. Found square shapes. Consider recoding.");
        # else:
            # cellPtIds           = [cellPointIds.GetId(i) for i in              \
            #                        xrange(cellPointIds.GetNumberOfIds())];
            # cellPtIds           = numpy.array(cellPtIds).astype(numpy.int64);
            # cellPtIds           = [];
            # cellPtIds.append(cellPointIds.GetId(0));
            # cellPtIds.append(cellPointIds.GetId(1));
            # cellPtIds.append(cellPointIds.GetId(2));
            # cellPtIds           = numpy.array(cellPtIds).astype(numpy.int64);




# def compute_boundary(polyData):
#     boundaryExtractor           = vtk.vtkFeatureEdges();

#     boundaryExtractor.SetInputData(polyData);
#     boundaryExtractor.BoundaryEdgesOn();
#     boundaryExtractor.FeatureEdgesOff();
#     boundaryExtractor.ManifoldEdgesOff();
#     boundaryExtractor.NonManifoldEdgesOff();
#     boundaryExtractor.Update();

#     polyDataEdges               = boundaryExtractor.GetOutput();

#     edgeIdDict                  = dict();
#     # edgeIdList                  = [];
#     # edgeIdSet                   = set();

#     for edge in xrange(polyDataEdges.GetNumberOfPoints()):
#         edgeIdDict[polyData.FindPoint(polyDataEdges.GetPoint(edge))] = polyDataEdges.GetPoint(edge);
#         # print(polyData.FindPoint(polyDataEdges.GetPoint(edge)));
#         # edgeIdList.append(polyData.FindPoint(polyDataEdges.GetPoint(edge)));
#         # edgeIdSet.add(polyData.FindPoint(polyDataEdges.GetPoint(edge)));

#     # boundary = order_boundary(polyData, edgeIdDict);

#     # return edgeIdList;
#     return edgeIdDict;
#     # return edgeIdSet;
#     # return boundary;






path            = os.path.join("/home/bee/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

polyData        = reader_vtk(path);
point_data      = extract_points(polyData);
polygon_data    = extract_polygons(polyData);


def compute_boundary(polyData):
    startingPoint                               = None;
    currentPoint                                = None;
    foundBoundary                               = False;
    cellId                                      = 0;
    boundary                                    = [];
    # boundary                                    = dict();
    visitedEdges                                = [];
    visitedPoints                               = [];
    visitedBoundaryEdges                        = [];

    for cellId in xrange(polyData.GetNumberOfCells()):
        cellPointIdList                         = vtk.vtkIdList();
        cellEdges                               = [];
        # vtkTriangle                             = polyData.GetCell(cellId);

        try:
            a = polyData.GetCellPoints(cellId, cellPointIdList);
        except:
            print("Cell ID does not exist. Possibly mesh provided is empty.");

        try:
            cellEdges = [[cellPointIdList.GetId(0), cellPointIdList.GetId(1)], \
                         [cellPointIdList.GetId(1), cellPointIdList.GetId(2)], \
                         [cellPointIdList.GetId(2), cellPointIdList.GetId(0)]];
        except:
            print("Mesh does not contain a triangulation. Check input.");

        for i in xrange(len(cellEdges)):
            if (cellEdges[i] in visitedEdges)   == False:
                visitedEdges.append(cellEdges[i]);

                edgeIdList                      = vtk.vtkIdList()
                a = edgeIdList.InsertNextId(cellEdges[i][0]);
                a = edgeIdList.InsertNextId(cellEdges[i][1]);

                singleCellEdgeNeighborIds       = vtk.vtkIdList();

                a = polyData.GetCellEdgeNeighbors(cellId,                      \
                                                  cellEdges[i][0],             \
                                                  cellEdges[i][1],             \
                                                  singleCellEdgeNeighborIds);

                if singleCellEdgeNeighborIds.GetNumberOfIds() == 0:
                    foundBoundary               = True;

                    boundary.append(cellEdges[i][0]);
                    boundary.append(cellEdges[i][1]);
                    # boundary[cellEdges[i][0]]   = polyData.GetPoint(cellEdges[i][0]);
                    # boundary[cellEdges[i][1]]   = polyData.GetPoint(cellEdges[i][1]);
                    startingPoint               = cellEdges[i][0];
                    currentPoint                = cellEdges[i][1];
                    visitedBoundaryEdges.append([currentPoint,startingPoint]);
                    visitedBoundaryEdges.append([startingPoint,currentPoint]);

                    # print(visitedBoundaryEdges);
                    # visitedBoundaryEdges        = [[startingPoint,currentPoint],
                    #                                [currentPoint,startingPoint]];

        if foundBoundary == True:
            break;

    if foundBoundary == False:
        raise Exception("The mesh provided has no boundary; not possible to do \
                         Quasi-Conformal Mapping on this dataset.");

    while currentPoint != startingPoint:
        neighboringCells                        = vtk.vtkIdList();

        polyData.GetPointCells(currentPoint, neighboringCells);

        for i in xrange(neighboringCells.GetNumberOfIds()):
            cell                                = neighboringCells.GetId(i);
            triangle                            = polyData.GetCell(cell);

            for j in xrange(triangle.GetNumberOfPoints()):
                # print("j: " + str(j));
                if triangle.GetPointId(j) == currentPoint:
                    j1                          = (j + 1) % 3;
                    j2                          = (j + 2) % 3;

                    edge1                       = [triangle.GetPointId(j),
                                                   triangle.GetPointId(j1)];
                    edge2                       = [triangle.GetPointId(j),
                                                   triangle.GetPointId(j2)];

            edgeNeighbors1                      = vtk.vtkIdList();
            edgeNeighbors2                      = vtk.vtkIdList();

            a = polyData.GetCellEdgeNeighbors(cell, edge1[0], edge1[1],    \
                                              edgeNeighbors1);

            a = polyData.GetCellEdgeNeighbors(cell, edge2[0], edge2[1],    \
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

            # if edgeNeighbors1.GetNumberOfIds() == 0:
            #     # print("1!!!!!!!!!!!!!!")
            #     if ([edge1[1], edge1[0]] in visitedBoundaryEdges) == False:
            #         # print([edge1[1], edge1[0]]);
            #         visitedBoundaryEdges.append([edge1[0], edge1[1]])
            #         visitedBoundaryEdges.append([edge1[1], edge1[0]])
            #         boundary.append(edge1[1]);
            #         # boundary[edge1[1]]          = polyData.GetPoint(edge1[1]);
            #         currentPoint                = edge1[1];
            #         break;

            # if edgeNeighbors2.GetNumberOfIds() == 0:
            #     # print("2!!!!!!!!!!!!!!");
            #     # print("visitedBoundaryEdges:");
            #     # print(visitedBoundaryEdges);

            #     if ([edge2[1], edge2[0]] in visitedBoundaryEdges) == False:
            #         # print([edge1[1], edge1[0]]);
            #         visitedBoundaryEdges.append([edge2[0], edge2[1]])
            #         visitedBoundaryEdges.append([edge2[1], edge2[0]])
            #         # print(""); print("visitedBoundaryEdges:"); print(visitedBoundaryEdges);

            #         boundary.append(edge2[1]);
            #         # boundary[edge2[1]]          = polyData.GetPoint(edge2[1]);
            #         currentPoint                = edge2[1];
            #         break;

    return boundary;


A = compute_boundary(polyData);


                    # edge3                       = [triangle.GetPointId(j),
                    #                                triangle.GetPointId(1)];

            # edge1                               = [triangle.GetEdge(0).GetPointId(0),
            #                                        triangle.GetEdge(0).GetPointId(1)];
            # edge2                               = [triangle.GetEdge(1).GetPointId(0),
            #                                        triangle.GetEdge(1).GetPointId(1)];
            # edge3                               = [triangle.GetEdge(2).GetPointId(0),
            #                                        triangle.GetEdge(2).GetPointId(1)];
            # edge2                               = triangle.GetEdge(1);
            # edge3                               = triangle.GetEdge(2);






cellId = 17 # a vtkQuad
cellPtIds = vtk.vtkIdList();
polyData.GetCellPoints(cellId, cellPtIds);
cellPtIds = [cellPtIds.GetId(i) for i in xrange(cellPtIds.GetNumberOfIds())];
cellPtIds = numpy.array(cellPtIds).astype(numpy.int64);
cellEdges = [[id0, id1] for id0, id1 in zip(cellPtIds[:-1], cellPtIds[1:])];
cellEdges.append([cellPtIds[-1], cellPtIds[0]]);

edgeId = 0 # or 1,2,3
edge = cellEdges[edgeId]

edgeIdList = vtk.vtkIdList()
for i in xrange(2):
    edgeIdList.InsertNextId(edge[i])

singleCellEdgeNeighborIds = vtk.vtkIdList()

polyData.GetCellEdgeNeighbors(cellId, edge[0], edge[1], singleCellEdgeNeighborIds)
polyData.GetCellNeighbors(cellId, edgeIdList, singleCellEdgeNeighborIds)



    while len(edgeIdDictOrdered)                != len(edgeIdDict):
        # print(len(edgeIdDictOrdered));
        visitedPoints.append(exploredPoint);
        neighboringPoints                      = [];
        neighboringCells                       = vtk.vtkIdList();

        polyData.GetPointCells(exploredPoint, neighboringCells);

        for i in xrange(neighboringCells.GetNumberOfIds()):
            cell                                = neighboringCells.GetId(i);

            if (polyData.GetCell(cell).GetPointId(0) in neighboringPoints) == False:
                neighboringPoints.append(polyData.GetCell(cell).GetPointId(0));

            if (polyData.GetCell(cell).GetPointId(1) in neighboringPoints) == False:
                neighboringPoints.append(polyData.GetCell(cell).GetPointId(1));

            if (polyData.GetCell(cell).GetPointId(2) in neighboringPoints) == False:
                neighboringPoints.append(polyData.GetCell(cell).GetPointId(2));

        for point in neighboringPoints:
            if (point in visitedPoints)         == False:
                if point in boundaryPoints:
                    edgeIdDictOrdered[point]    = edgeIdDict[point];
                    exploredPoint               = point;
                    break;






# def compute_boundary(polyData):
#     boundaryExtractor           = vtk.vtkFeatureEdges();

#     boundaryExtractor.SetInputData(polyData);
#     boundaryExtractor.BoundaryEdgesOn();
#     boundaryExtractor.FeatureEdgesOff();
#     boundaryExtractor.ManifoldEdgesOff();
#     boundaryExtractor.NonManifoldEdgesOff();
#     boundaryExtractor.Update();

#     polyDataEdges               = boundaryExtractor.GetOutput();

#     edgeIdDict                  = dict();
#     # edgeIdList                  = [];
#     # edgeIdSet                   = set();

#     for edge in xrange(polyDataEdges.GetNumberOfPoints()):
#         edgeIdDict[polyData.FindPoint(polyDataEdges.GetPoint(edge))] = polyDataEdges.GetPoint(edge);
#         # print(polyData.FindPoint(polyDataEdges.GetPoint(edge)));
#         # edgeIdList.append(polyData.FindPoint(polyDataEdges.GetPoint(edge)));
#         # edgeIdSet.add(polyData.FindPoint(polyDataEdges.GetPoint(edge)));

#     # boundary = order_boundary(polyData, edgeIdDict);

#     # return edgeIdList;
#     return edgeIdDict;
#     # return edgeIdSet;
#     # return boundary;



# def order_boundary(polyData, edgeIdDict):
#     edgeIdDictOrdered                           = collections.defaultdict();
#     boundaryPoints                              = edgeIdDict.keys();
#     visitedPoints                               = [];
#     startingPoint                               = None;
#     exploredPoint                               = None;

#     try:
#         startingPoint                           = boundaryPoints[0];
#         exploredPoint                           = startingPoint;
#         edgeIdDictOrdered[startingPoint]        = edgeIdDict[startingPoint];
#     except:
#         print("Size of keyring is zero; no lines found in provided mesh.");

#     while len(edgeIdDictOrdered)                != len(edgeIdDict):
#         # print(len(edgeIdDictOrdered));
#         visitedPoints.append(exploredPoint);
#         neighboringPoints                      = [];
#         neighboringCells                       = vtk.vtkIdList();

#         polyData.GetPointCells(exploredPoint, neighboringCells);

#         for i in xrange(neighboringCells.GetNumberOfIds()):
#             cell                                = neighboringCells.GetId(i);

#             if (polyData.GetCell(cell).GetPointId(0) in neighboringPoints) == False:
#                 neighboringPoints.append(polyData.GetCell(cell).GetPointId(0));

#             if (polyData.GetCell(cell).GetPointId(1) in neighboringPoints) == False:
#                 neighboringPoints.append(polyData.GetCell(cell).GetPointId(1));

#             if (polyData.GetCell(cell).GetPointId(2) in neighboringPoints) == False:
#                 neighboringPoints.append(polyData.GetCell(cell).GetPointId(2));

#         for point in neighboringPoints:
#             if (point in visitedPoints)         == False:
#                 if point in boundaryPoints:
#                     edgeIdDictOrdered[point]    = edgeIdDict[point];
#                     exploredPoint               = point;
#                     break;

#     return edgeIdDictOrdered;







# path      = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");
path            = os.path.join("/home/bee/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

polyData        = reader_vtk(path);
point_data      = extract_points(polyData);
polygon_data    = extract_polygons(polyData);

bound           = compute_boundary(polyData);
boundOrdered    = order_boundary(polyData, bound);



    # while len(edgeIdDictOrdered)                != len(edgeIdDict):
    #     if (exploredPoint in visitedPoints)     == False:
    #         visitedPoints.append(exploredPoint);

    #         neighboringCells                   = vtk.vtkIdList();
    #         polyData.GetPointCells(startingPoint, neighboringCells);

    #         for i in xrange(neighboringCells.GetNumberOfIds()):
    #             cell                            = neighboringCells.GetId(i);



    #             if 

    #             if polyData.GetCell(cell).GetPointId(0) in boundaryPoints:


    #         exploredPoint                       = nextPoint;
    #     else:
    #         print("Check it. Something happened.");

    # return edgeIdDictOrdered;












    


# cellToSearch = cellsOfPoint.GetId(1); cellToSearch; print(""); cell = polyData.GetCell(cellToSearch); cell.GetPointId(0); cell.GetPointId(1); cell.GetPointId(2);





    # start = time.time();
    # for cellId in xrange(polyData.GetNumberOfCells()):
    #     cellPointIds            = vtk.vtkIdList();

    #     try:
    #         triangle            = polyData.GetCell(cellId);
    #         # polyData.GetCellPoints(cellId, cellPointIds);
    #     except:
    #         print("Tried to access a cell ID that is non-existant. Re-code \
    #                compute_boundary function. It seems that cell IDs are not \
    #                always consecutive as initially thought.");

    #     edge1                   = triangle.GetEdge(0);
    #     edge2                   = triangle.GetEdge(1);
    #     edge3                   = triangle.GetEdge(2);

    # print(time.time() - start);



# extractor = vtk.vtkExtractEdges();
# extractor.SetInputData(polyData);
# extractor.Update()


# start = time.time(); 
# extractor222 = vtk.vtkExtractEdges(); 
# extractor222.SetInputData(polyData); 
# extractor222.Update(); 
# print(time.time() - start);



# cellId = 17 # a vtkQuad
# cellPtIds = vtk.vtkIdList();
# polyData.GetCellPoints(cellId, cellPtIds);
# cellPtIds = [cellPtIds.GetId(i) for i in xrange(cellPtIds.GetNumberOfIds())];
# cellPtIds = numpy.array(cellPtIds).astype(numpy.int64);
# cellEdges = [[id0, id1] for id0, id1 in zip(cellPtIds[:-1], cellPtIds[1:])];
# cellEdges.append([cellPtIds[-1], cellPtIds[0]]);

# edgeId = 0 # or 1,2,3
# edge = cellEdges[edgeId]

# edgeIdList = vtk.vtkIdList()
# for i in xrange(2):
#     edgeIdList.InsertNextId(edge[i])

# singleCellEdgeNeighborIds = vtk.vtkIdList()
# polyData.GetCellEdgeNeighbors(cellId, edge[0], edge[1], singleCellEdgeNeighborIds)
# polyData.GetCellNeighbors(cellId, edgeIdList, singleCellEdgeNeighborIds)

# >>> triangle = polyData.GetCell(singleCellEdgeNeighborIds.GetId(0))
# >>> edge
# [15, 17]
# >>> print(triangle.GetPointId(0))
# 17
# >>> print(triangle.GetPointId(1))
# 15
# >>> print(triangle.GetPointId(2))
# 382


def print_it(bound, polyData):
    connected = True;
    # print("Points of line:"); 

    for i in range(0, bound.GetNumberOfPoints()):
        connected_neighbour = False;
        point_actual = polyData.FindPoint(bound.GetPoint(i)); 

        if i != bound.GetNumberOfPoints() - 1:
            point_next = polyData.FindPoint(bound.GetPoint(i + 1));
        else:
            point_next = polyData.FindPoint(bound.GetPoint(0));

        cellsOfPoint = vtk.vtkIdList(); 
        polyData.GetPointCells(point_actual, cellsOfPoint); 

        print("\nPoint IDs:"); 
        print([point_actual, point_next]); 

        for j in xrange(cellsOfPoint.GetNumberOfIds()):
            if connected_neighbour == False:
                cellToSearch = cellsOfPoint.GetId(j);

                cell = polyData.GetCell(cellToSearch);
    
                print("Point IDs of neighboring cells:");
                print(cell.GetPointId(0));
                print(cell.GetPointId(1));
                print(cell.GetPointId(2));

                if cell.GetPointId(0) == point_next:
                    connected_neighbour = True;
                elif cell.GetPointId(1) == point_next:
                    connected_neighbour = True;
                elif cell.GetPointId(2) == point_next:
                    connected_neighbour = True;

        connected = (connected and connected_neighbour);

    return connected;




        # cellToSearch = cellsOfPoint.GetId(1);

        # cellToSearch; print("");

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










