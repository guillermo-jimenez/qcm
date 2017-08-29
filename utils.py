# -*- coding: utf-8 -*-

"""
    Copyright (C) 2017 - Universitat Pompeu Fabra
    Authors - Guillermo Jimenez-Perez <guillermo.jim.per@gmail.com>
            - Constantine Butakoff 

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

from os.path import isfile
from os.path import splitext
from os.path import split
from os.path import isdir
from os.path import join

from vtk import vtkPolyData
from vtk import vtkPolyDataReader
from vtk import vtkPoints
from vtk import vtkCellArray
from vtk import vtkDoubleArray
from vtk import vtkTriangle
from vtk import vtkIdList
from vtk import vtkClipPolyData
from vtk import vtkPlane

from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy import mean
from numpy import cross
from numpy import dot
from numpy import where

from scipy import sqrt
from scipy import arccos
from scipy import tan
from scipy import flipud
from scipy import roll

from scipy.linalg import norm

from scipy.sparse import csr_matrix
from scipy.sparse import triu
from scipy.sparse import find

from numpy.matlib import repmat

import vtk
import time
import numpy

from vtk import vtkCommand
from vtk import vtkSphereSource
from vtk import vtkPolyDataMapper
from vtk import vtkActor
from vtk import vtkPoints
from vtk import vtkIdList
from vtk import vtkPolyDataMapper
from vtk import vtkRenderer
from vtk import vtkRenderWindow
from vtk import vtkRenderWindowInteractor
from vtk import vtkInteractorStyleTrackballCamera
from vtk import vtkPointPicker



class PointPicker(vtkPointPicker):
    def __init__(self,parent=None):
        self.AddObserver(vtkCommand.EndPickEvent, self.EndPickEvent)

    #these are the variables te user will set in the PointSelector class. Here we just take the pointers
    def SetParameters(self, selected_points, selected_point_ids, marker_radius, marker_colors):
        self.selected_points = selected_points
        self.marker_radius = marker_radius
        self.marker_colors = marker_colors
        self.selected_point_ids = selected_point_ids
        
    #callback after every picking event    
    def EndPickEvent(self,obj,event):
        rnd = self.GetRenderer()  

        n_points = self.selected_points.GetNumberOfPoints();
        
        #check if anything was picked
        pt_id = self.GetPointId()
        if pt_id >= 0:
            if n_points < len(self.marker_colors):
                #create a sphere to mark the location
                sphereSource = vtkSphereSource();
                sphereSource.SetRadius(self.marker_radius); 
                sphereSource.SetCenter(self.GetPickPosition());        
                
                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(sphereSource.GetOutputPort())

                actor = vtkActor()
                actor.SetMapper(mapper)

                #define the color of the sphere (pick from the list)
                actor.GetProperty().SetColor(self.marker_colors[n_points])
                rnd.AddActor(actor)

                #populate the list of ids and coordinates
                self.selected_points.InsertNextPoint(self.GetPickPosition())
                self.selected_point_ids.InsertNextId(pt_id)
            

#the main class
class PointSelector:

    def __init__(self, pointIds=None): #initialize variables
        self.marker_radius      = 1
        self.marker_colors      = [(1,0,0), (0,1,0), (1,1,0), (0,0,0), (0.5,0.5,0.5), (0.5,0,0)] #different colors for different markers
        # self.marker_colors      = [(1,0,0), (0,1,0), (1,1,0), (0,0,0), (0.5,1,0.5)] #different colors for different markers
        self.selected_points    = vtkPoints()
        self.selected_point_ids = vtkIdList()
        self.window_size        = (800,600)
        self.pointIds           = pointIds

    def GetSelectedPointIds(self): #returns vtkIdList in the order of clicks
        return self.selected_point_ids
        
    def GetSelectedPoints(self): #returns vtkPoints in the order of clicks
        return self.selected_points
        
    def DoSelection(self, shape): #open rendering window and start 
        if self.pointIds is None:
            self.selected_points.Reset()
            self.selected_point_ids.Reset()
        else:
            try:
                for i in range(0, len(self.pointIds)):
                    self.selected_point_ids.InsertNextId(self.pointIds[i])
                    self.selected_points.InsertNextPoint(shape.GetPoint(self.pointIds[i]))
            except:
                raise Exception("pointIds has to be iterable")

            renderer = vtkRenderer();

            #check if anything was picked
            for i in range(0, self.selected_points.GetNumberOfPoints()):
                if i < len(self.marker_colors):
                    #create a sphere to mark the location
                    sphereSource = vtkSphereSource();
                    sphereSource.SetRadius(self.marker_radius); 
                    sphereSource.SetCenter(self.selected_points.GetPoint(i));
                    
                    mapper = vtkPolyDataMapper()
                    mapper.SetInputConnection(sphereSource.GetOutputPort())

                    actor = vtkActor()
                    actor.SetMapper(mapper)

                    #define the color of the sphere (pick from the list)
                    actor.GetProperty().SetColor(self.marker_colors[i])

                    renderer.AddActor(actor)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(shape)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(1)

        pointPicker = PointPicker()
        pointPicker.AddPickList(actor)
        pointPicker.PickFromListOn()

        pointPicker.SetParameters(self.selected_points, self.selected_point_ids, self.marker_radius, self.marker_colors)

        try:
            renderer.AddActor(actor)
        except:
            renderer = vtkRenderer();
            renderer.AddActor(actor)

        window = vtkRenderWindow();
        window.AddRenderer( renderer );

        interactor = vtkRenderWindowInteractor();
        interactor.SetRenderWindow( window );

        interactor_style = vtkInteractorStyleTrackballCamera() 
        interactor.SetInteractorStyle( interactor_style )
        interactor.SetPicker(pointPicker); 

        window.SetSize(self.window_size)
        window.Render()
        interactor.Start();

        render_window = interactor.GetRenderWindow()
        render_window.Finalize()



def cartoReader(path, output_path=None, tags=None, write=False):
    """ returns a polydata.

    option write as True creates a .vtk file with the polydata

    tags need to be in the shape of a dict of tags, equal to False:
    tags                                = dict()

    tags['[NameOfTheTag]']              = False

    Example:
    tags['[VerticesSection]']           = False
    tags['[TrianglesSection]']          = False
    tags['[VerticesColorsSection]']     = False
    tags['[VerticesAttributesSection]'] = False

    """

    print("TO-DO: DOCUMENTATION")

    # Initializing auxiliary variables
    polydata        = vtkPolyData()

    attributes      = dict()

    points          = vtkPoints()
    cells           = vtkCellArray()

    pointNormals    = vtkDoubleArray()
    cellNormals     = vtkDoubleArray()
    cellArray       = vtkDoubleArray()

    signals         = []
    boolSignals     = False

    pointNormals.SetNumberOfComponents(3)
    pointNormals.SetNumberOfTuples(points.GetNumberOfPoints())
    pointNormals.SetName('PointNormals')

    cellNormals.SetNumberOfComponents(3)
    cellNormals.SetNumberOfTuples(points.GetNumberOfPoints())
    cellNormals.SetName('CellNormals')


    # Checking the input and output directions
    if not isfile(path):
        raise RuntimeError("File does not exist")

    if (splitext(path)[1] != '.mesh'):
        raise Exception("Only '.mesh' files are accepted. Set a new file path.")

    if output_path is None:
        print(" * Writing to default location: ")

        directory, filename = split(path)
        filename, extension = splitext(filename)
        
        if isdir(join(directory, 'VTK')):
            output_path     = join(directory, 'VTK', str(filename + '.vtk'))
        else:
            mkdir(join(directory, 'VTK'))

            if isdir(join(directory, 'VTK')):
                output_path = join(directory, 'VTK', str(filename + '.vtk'))
            else:
                print("  !-> Could not create output directory. Writing in input directory")
                output_path = join(directory, str(filename + '.vtk'))

        print("  --> " + output_path + "\n")


    # Check the tags variable
    if tags is None:
        print(" * Provided no tag specification. Using default...\n")

        tags                                = dict()

        tags['[GeneralAttributes]']         = False
        tags['[VerticesSection]']           = False
        tags['[TrianglesSection]']          = False
        tags['[VerticesColorsSection]']     = False
        tags['[VerticesAttributesSection]'] = False

    # Analyze the input file
    for i in range(0, len(data)):
        # Blank space
        if data[i] == '':
            continue

        # Comment symbol in .mesh files
        elif ((data[i][0] == '#') or (data[i][0] == ';')):
            continue

        # Starting character for tags. Activate the tag and deactivate others
        elif data[i][0] == '[':
            for j in tags:
                tags[j]          = False

            if data[i] in tags:
                tags[data[i]]    = True
                continue

            else:
                raise RuntimeError('Tag not considered. Contact maintainer.\nLine states: ' + data[i])

        # If tag [GeneralAttributes] is active...
        if tags['[GeneralAttributes]']:
            splits = data[i].split()
            attributes[splits[0]] = [splits[i] for i in range(2, len(splits))]

        # If tag [GeneralAttributes] has been just deactivated...
        elif boolSignals == False:
            boolSignals = True

            nPoints      = int(attributes['NumVertex'][0])
            nColors      = int(attributes['NumVertexColors'][0])
            nAttrib      = int(attributes['NumVertexAttributes'][0])

            for j in range(0, (nColors + nAttrib)):
                cellArray = vtkDoubleArray()
                cellArray.SetNumberOfComponents(1)

                if j < nColors:
                    cellArray.SetName(attributes['ColorsNames'][j])
                else:
                    cellArray.SetName(attributes['VertexAttributesNames'][j - nColors])

                signals.append(cellArray)

        # If tag [VerticesSection] is active...
        if tags['[VerticesSection]']:
            points.InsertPoint(int(data[i].split()[0]), (float(data[i].split()[2]),
                                                         float(data[i].split()[3]),
                                                         float(data[i].split()[4])))

            pointNormals.InsertNextTuple3(float(data[i].split()[5]),
                                          float(data[i].split()[6]),
                                          float(data[i].split()[7]))

            ### FALTA EL GROUPID DE ESTE VERTICESSECTION

        # If tag [TrianglesSection] is active...
        elif tags['[TrianglesSection]']:
            triangle = vtkTriangle()

            triangle.GetPointIds().SetId(0, int(data[i].split()[2]))
            triangle.GetPointIds().SetId(1, int(data[i].split()[3]))
            triangle.GetPointIds().SetId(2, int(data[i].split()[4]))

            cells.InsertNextCell(triangle)

            cellNormals.InsertNextTuple3(float(data[i].split()[5]),
                                         float(data[i].split()[6]),
                                         float(data[i].split()[7]))

            ### FALTA EL GROUPID DE ESTE TRIANGLESSECTION
                
        # If tag [VerticesColorsSection] is active...
        elif tags['[VerticesColorsSection]']:
            for j in range(2, len(data[i].split())):
                signals[j-2].InsertNextTuple1(float(data[i].split()[j]))

        # If tag [VerticesAttributesSection] is active...
        elif tags['[VerticesAttributesSection]']:
            for j in range(2, len(data[i].split())):
                signals[nColors + j - 2].InsertNextTuple1(float(data[i].split()[j]))

    # Fill the polydata variable
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    polydata.GetPointData().SetNormals(pointNormals)
    polydata.GetCellData().SetNormals(cellNormals)

    for array in signals:
        if array.GetName() == 'Bipolar':
            polydata.GetPointData().SetScalars(array)
        else:
            polydata.GetPointData().AddArray(array)

    # If function is called with 'write' set to True (default, False)...
    if write == True:
        writer = vtkPolyDataWriter()
        writer.SetFileName(output_path)

        writer.SetInputData(polydata)
        writer.SetFileName(output_path)
        writer.Write()

    return polydata



def polydataReader(path):
    """ """

    print("TO-DO: DOCUMENTATION")

    # Reading the input VTK file
    if (path is not None):
        if not isfile(path):
            raise RuntimeError("File does not exist")

        reader                  = vtkPolyDataReader()
        reader.SetFileName(path)
        reader.Update()

        polydata         = reader.GetOutput()
        polydata.BuildLinks()

    return polydata


def vtkPointsToNumpy(polydata):
    try:
        pointVector             = polydata.GetPoints()
    except:
        raise Exception("The input provided is not a VTK polydata variable")

    try:
        rows                    = len(pointVector.GetPoint(0))
        cols                    = pointVector.GetNumberOfPoints()
        points                  = zeros((rows,cols))
    except:
        raise Exception("The VTK file provided does not contain any points")

    if pointVector:
        for i in range(0, pointVector.GetNumberOfPoints()):
            point_tuple         = pointVector.GetPoint(i)

            points[0,i]         = point_tuple[0]
            points[1,i]         = point_tuple[1]
            points[2,i]         = point_tuple[2]

    return points


def vtkCellsToNumpy(polydata):
    try:
        pointVector             = polydata.GetPoints()
    except:
        raise Exception("The input provided is not a VTK polydata variable")

    polygons                    = None

    for i in range(0, polydata.GetNumberOfCells()):
        pointIds                = polydata.GetCell(i).GetPointIds()

        if polygons is None:
            try:
                rows            = pointIds.GetNumberOfIds()
                cols            = polydata.GetNumberOfCells()
                polygons        = zeros((rows,cols), dtype=int)
            except:
                raise Exception("The VTK file provided does not contain a triangulation")

        polygons[0,i]           = pointIds.GetId(0)
        polygons[1,i]           = pointIds.GetId(1)
        polygons[2,i]           = pointIds.GetId(2)

    return polygons


def adjacencyMatrix(polydata, polygons=None):
    """ """

    print("TO-DO: DOCUMENTATION")

    if polygons is None:
        polygons        = vtkCellsToNumpy(polydata)

    numPoints           = polydata.GetNumberOfPoints()
    numPolygons         = polydata.GetNumberOfPolys()
    numDims             = polydata.GetPoints().GetData().GetNumberOfComponents()
    adjMatrix           = csr_matrix((numPoints, numPoints))
    
    for i in range(0, numDims):
        i1              = (i + 0)%3
        i2              = (i + 1)%3
        i3              = (i + 2)%3
        
        iterData1       = csr_matrix((ones(numPolygons,), (polygons[i2,:], polygons[i3,:])), shape=(numPoints, numPoints))
        iterData2       = csr_matrix((ones(numPolygons,), (polygons[i3,:], polygons[i2,:])), shape=(numPoints, numPoints))

        adjMatrix       = adjMatrix + iterData1 + iterData2

    return adjMatrix


def cotangentWeightsLaplacianMatrix(polydata, points=None, polygons=None):
    """ """

    print("TO-DO: DOCUMENTATION")

    if points is None:
        points          = vtkPointsToNumpy(polydata)

    if polygons is None:
        polygons        = vtkCellsToNumpy(polydata)

    numPoints           = polydata.GetNumberOfPoints()
    numPolygons         = polydata.GetNumberOfPolys()
    numDims             = polydata.GetPoints().GetData().GetNumberOfComponents()
    laplacianMatrix     = csr_matrix((numPoints, numPoints))

    for i in range(0, numDims):
        i1              = (i + 0)%3
        i2              = (i + 1)%3
        i3              = (i + 2)%3

        vectP2P1        = (points[:, polygons[i2, :]] - points[:, polygons[i1, :]])
        vectP3P1        = (points[:, polygons[i3, :]] - points[:, polygons[i1, :]])

        vectP2P1        = vectP2P1 / repmat(sqrt((vectP2P1**2).sum(0)), numDims, 1)
        vectP3P1        = vectP3P1 / repmat(sqrt((vectP3P1**2).sum(0)), numDims, 1)

        angles          = arccos((vectP2P1 * vectP3P1).sum(0))

        iterData1       = csr_matrix((1/tan(angles), (polygons[i2,:], polygons[i3,:])), 
                                     shape=(numPoints, numPoints))

        iterData2       = csr_matrix((1/tan(angles), (polygons[i3,:], polygons[i2,:])), 
                                     shape=(numPoints, numPoints))

        laplacianMatrix = laplacianMatrix + iterData1 + iterData2

    return laplacianMatrix


def boundaryExtractor(polydata, polygons=None, adjMatrix=None):
    """ """

    print("TO-DO: DOCUMENTATION")

    if polygons is None:
        polygons    = vtkCellsToNumpy(polydata)

    if adjMatrix is None:
        adjMatrix   = adjacencyMatrix(polydata, polygons)

    # If the adjacency matrix shows edges with only one adjacent cell, the
    # edge is part of the boundary
    preboundary     = find(adjMatrix==1)
    preboundary     = asarray([preboundary[0].tolist(), preboundary[1].tolist()])

    if preboundary[0].shape[0] == 0:
        raise Exception("Shape with no boundary has been found. " + \
                        "Run vtkClippingPlane to produce a boundary.") 

    counter         = 0
    edgesUsed       = 0
    visited         = []
    boundary        = []

    # Check for multiple boundaries
    while(edgesUsed != int(preboundary.shape[1]/2)):
        start           = preboundary[0,counter]

        if (start in visited) == False:
            current         = start
            nxt             = preboundary[1,counter]
            invalid         = [(nxt, current)]
            boundaryAux     = [current]
            visited.append(current)

            # Recognize a circular path from a point to itself
            while(nxt != start):
                indices = where(preboundary[0,:]==nxt)[0]

                edge1       = preboundary[:,indices[0]]
                edge2       = preboundary[:,indices[1]]

                # Check if the edge has already been visited in the opposite
                # direction. If it has, the other edge is the valid edge
                if (edge1[0], edge1[1]) in invalid:
                    current = edge2[0]
                    nxt     = edge2[1]
                    invalid.append((nxt, current))
                    boundaryAux.append(current)
                    visited.append(current)
                    
                # Check if the edge has already been visited in the opposite
                # direction. If it has, the other edge is the valid edge
                elif (edge2[0], edge2[1]) in invalid:
                    current = edge1[0]
                    nxt     = edge1[1]
                    invalid.append((nxt, current))
                    boundaryAux.append(current)
                    visited.append(current)

            # Calculate the total number of edges assigned to a boundary
            edgesUsed = edgesUsed + len(boundaryAux)
            boundaryAux = asarray(boundaryAux)
            boundary.append(boundaryAux)

        # For multiple boundaries, check every vertex as possible boundary starter
        counter = counter + 1

    return boundary


def landmarkSelector(polydata, totalLandmarks, landmarks=None):
    """ """

    print("TO-DO: DOCUMENTATION")

    if landmarks is None:
        landmarks = []

    if len(landmarks) < totalLandmarks:
        ps = PointSelector(landmarks)

        while(ps.selected_point_ids.GetNumberOfIds() != totalLandmarks):
            ps = PointSelector(landmarks)
            ps.DoSelection(polydata)

        for i in range(0, ps.GetSelectedPointIds().GetNumberOfIds()):
            try:
                landmarksSelected.append(ps.GetSelectedPointIds().GetId(i))
            except:
                landmarksSelected   = [ps.GetSelectedPointIds().GetId(i)]

        return landmarksSelected

    else:
        return landmarks


def vtkClippingPlane(polydata, landmarks=None, reverse=False):
    """ """

    print("TO-DO: DOCUMENTATION")

    plane       = vtkPlane()
    clip        = vtkClipPolyData()
    landmarks   = landmarkSelector(polydata, 3, landmarks)

    O           = asarray(polydata.GetPoint(landmarks[0]))
    A           = asarray(polydata.GetPoint(landmarks[1]))
    B           = asarray(polydata.GetPoint(landmarks[2]))

    OA          = A - O
    OB          = B - O

    normal      = cross(OB,OA)/norm(cross(OA,OB))

    plane.SetOrigin(O)
    plane.SetNormal(tuple(normal))

    if reverse is True:
        clip.InsideOutOn()
    else:
        clip.InsideOutOff()

    clip.SetClipFunction(plane)
    clip.SetInputData(polydata)
    clip.Update()
    
    return clip.GetOutput()    


def closestBoundaryId(polydata, objectivePointId, boundary=None, polygons=None, adjMatrix=None):
    """ """

    print("TO-DO: DOCUMENTATION")

    closestPointIndex       = None
    boundaryVector          = []
    boundaryNumber          = []
    totalBoundaryPoints     = 0
    # inBoundary              = False

    try:
        point               = polydata.GetPoint(objectivePointId)
    except:
        raise Exception("Mesh is empty or wrong objective point provided")

    # Calculate the boundary if not provided
    if boundary is None:
        if polygons is None:
            polygons        = vtkCellsToNumpy(polydata)

        if adjMatrix is None:
            adjMatrix       = adjacencyMatrix(polydata, polygons)

        boundary            = boundaryExtractor(polydata, polygons, adjMatrix)

    # For each of the boundaries:
    for i in range(0, len(boundary)):
        totalBoundaryPoints     = totalBoundaryPoints + boundary[i].size

        # Check if the objective point is already part of a boundary and return
        if (objectivePointId in boundary[i]) == True:
            closestPointIndex   = where(boundary[i]==objectivePointId)

            # If the index provided is contained in any boundary, mark it
            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0]
                    
                    return (i, closestPointIndex)
                else:
                    raise Exception("Mesh or boundary contains repeated point IDs")
            else:
                raise Exception("Mesh or boundary contains repeated point IDs")

        # else:
        #     inBoundary              = inBoundary and False

        # If not, calculate the distance of the point to every point of each 
        # boundary. The index with the global lowest distance is selected
        else:
            # Calculate the distance to each point of each boundary
            for j in range(0, len(boundary[i])):
                boundaryVector.append(polydata.GetPoint(boundary[i][j]))
                boundaryNumber.append(i)

    boundaryVector              = asarray(boundaryVector)
    pointVector                 = repmat(point, totalBoundaryPoints, 1)

    distanceToObjectivePoint    = (boundaryVector - pointVector)
    distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(1))
    closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

    if len(closestPointIndex) == 1:
        if len(closestPointIndex[0]) == 1:
            closestPointIndex   = closestPointIndex[0][0]
        else:
            raise Exception("Mesh or boundary contains repeated point IDs")
    else:
        raise Exception("Mesh or boundary contains repeated point IDs")

    return (boundaryNumber[closestPointIndex], closestPointIndex)



        # else:
        #     for i in range(0, len(boundary)):

        #     if len(boundary.shape) == 1:
        #         for i in boundary.shape[0]:
                    
        #         septalPoint     = repmat(septalPoint, self.boundary.size, 1)
        #         septalPoint     = septalPoint.transpose()
        #     else:
        #         for i in boundary.shape[1]:


        #     distanceToObjectivePoint    = (self.points[:, self.boundary] - septalPoint)
        #     distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0))
        #     closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min())

        #     if len(closestPointIndex) == 1:
        #         if len(closestPointIndex[0]) == 1:
        #             closestPointIndex   = closestPointIndex[0][0]
        #         else:
        #             raise Exception("Mesh or boundary contains repeated point IDs")
        #     else:
        #         raise Exception("Mesh or boundary contains repeated point IDs")

