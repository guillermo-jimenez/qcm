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

from os import system
from os import mkdir
from os.path import isfile
from os.path import splitext
from os.path import split
from os.path import isdir
from os.path import join

from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy import mean
from numpy import cross
from numpy import dot
from numpy import where
from numpy.matlib import repmat

from scipy import sqrt
from scipy import arccos
from scipy import tan
from scipy import flipud
from scipy import roll
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.sparse import triu
from scipy.sparse import find

from sklearn.neighbors import NearestNeighbors

from vtk import vtkPolyData
from vtk import vtkPolyDataReader
from vtk import vtkPoints
from vtk import vtkCellArray
from vtk import vtkDoubleArray
from vtk import vtkTriangle
from vtk import vtkClipPolyData
from vtk import vtkPlane
from vtk import vtkActor
from vtk import vtkFloatArray
from vtk import vtkPolyDataMapper
from vtk import vtkRenderer
from vtk import vtkRenderWindow
from vtk import vtkRenderWindowInteractor
from vtk import vtkInteractorStyleTrackballCamera
from vtk import vtkScalarBarActor
from vtk import vtkLookupTable

from PointPicker import PointPicker
from PointPicker import PointSelector



def cartoReader(path, output_path=None, tags=None, write=False):
    """Returns a vtkPolyData object from a '.mesh' input file.

    Notes:
        Argument 'tags' needs to be in the shape of a dict of tags, equal to False:
        tags                                = dict()

        tags['[NameOfTheTag]']              = False

        Example:
        tags['[VerticesSection]']           = False
        tags['[TrianglesSection]']          = False
        tags['[VerticesColorsSection]']     = False
        tags['[VerticesAttributesSection]'] = False


    Args:
        output_path (str): Absolute output path for writing a .vtk file.
        write (bool): If True, creates a .vtk file with the polydata.
        tags (dict): Tags employed in the .mesh file. See notes for correct specification.

    """

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
    """Reads a '.vtk' file

    Return:
        polydata (vtkPolyData): vtkPolyData object with the '.vtk' information.

    Args:
        path (str): Absolute path for the input '.vtk' file.
    """

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
    """Extracts the point information of a vtkPolyData object into a numpy array.
    The extracted coordinates data is ordered sequentially: the first row in the 
    array coincides with the first vertex in the polydata and so on.  

    Return:
        points (numpy.ndarray): numpy.ndarray object with the coordinate information.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh
    """

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
    """Extracts the vertices of each triangle of a vtkPolyData object into a 
    numpy array. The extracted information is coded so that in every column, the 
    identifiers of the vertices of each triangle is stored. 

    Return:
        polygons (numpy.ndarray): numpy.ndarray object with the triangles information

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh
    """

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
    """Calculates the adjacency matrix of the triangulation.

    Return:
        adjMatrix (scipy.sparse.csr.csr_matrix): adjacency matrix.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh
        polygons (numpy.ndarray): information on the identifiers of every point in every triangle of the mesh
    """

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
    """Calculates the laplacian matrix from the cotangent weights of the mesh.

    Return:
        laplacianMatrix (scipy.sparse.csr.csr_matrix): laplacian matrix.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh
        points (numpy.ndarray): information of the coordinates of each vertex
        polygons (numpy.ndarray): information on the identifiers of every point in every triangle of the mesh
    """

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
    """Extracts the boundary (sequences of vertices that are connected by edges
    that are not shared by more than one triangle) from the polydata. Polygons 
    and the adjacency matrix can be provided to accelerate calculations.

    The boundary is calculated with the adjacency matrix, searching for points
    that are not shared by more than one cell. After a list of vertices that 
    share that definition is extracted, it is further divided into its 
    connected components.

    Notes:
        The boundary returned takes the shape of a list of lists. Each element
        in the main list is a connected component, which is in turn a list of
        the specific vertices that make that connected component.

    Return:
        boundary (list): list of connected components.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh.
        polygons (numpy.ndarray): information on the identifiers of every point in every triangle of the mesh.
        adjMatrix (scipy.sparse.csr.csr_matrix): adjacency matrix.
    """

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
    """Launches a visualization window that allows for the visualization of the
    mesh and the selection of landmarks for their later usage.

    For selecting the points, click inside the visualization window once, then 
    drag and drop until the point-to-be-selected is clearly visible. When the
    point is clearly visualizable, hold the mouse still on top of the 
    aforementioned point and, while not moving, press 'p' on the keyboard.

    Repeat the operation as many times as needed, until the number of points
    selected is equal to the input variable 'totalLandmarks'. If the number of
    points provided is not equal to the number of landmarks, the point selection
    will start over.

    Return:
        landmarks (list): list of IDs of the vertices in the input polydata.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh.
        totalLandmarks (int): Number of landmarks to be extracted.
        landmarks (list): List comprising any previously selected landmark.
    """

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

    """Produces a new vtkPolyData from clipping the provided polydata using a 
    plane. If no points are provided to create the clipping plane (three points), 
    the 'landmarkSelector' function will be executed to acquire them.

    Return:
        clip (vtkPolyData): clipped polydata

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh.
        landmarks (list): list of int consisting of the IDs of the vertices to be used for the clipping plane.
        reverse (bool): Reverses the normal for the clipping plane, obtaining the opposite side of the clipping plane.
    """

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


def vtkVisualize(polydata, activeArray=0, visualizationRange=None):
    """Launches a visualization window that allows for the visualization of the
    mesh. The active scalar array visualized can be changed with the 
    'activeArray' input variable.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh.
        activeArray (int): Identifier of the scalar vector to be visualized.
        visualizationRange (tuple): Two-element tuple comprising the range of values to be visualized.
    """

    if polydata.GetPointData().GetNumberOfArrays() > 0:
        if activeArray < polydata.GetPointData().GetNumberOfArrays():
            polydata.GetPointData().SetActiveScalars(polydata.GetPointData().GetArray(activeArray).GetName())
        else:
            print("Selected array is non-existent. Selecting first array")
            polydata.GetPointData().SetActiveScalars(polydata.GetPointData().GetArray(0).GetName())

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    try:
        mapper.SetScalarRange(visualizationRange)
    except:
        mapper.SetScalarRange(polydata.GetPointData().GetScalars().GetValueRange())

    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToMapScalars()

    actor = vtkActor()
    actor.SetMapper(mapper)

    scalarBar = vtkScalarBarActor()
    scalarBar.SetLookupTable(mapper.GetLookupTable())
    scalarBar.SetTitle(polydata.GetPointData().GetScalars().GetName())
    scalarBar.SetNumberOfLabels(4)

    hueLut = vtkLookupTable()
    hueLut.Build()

    mapper.SetLookupTable(hueLut);
    scalarBar.SetLookupTable(hueLut);

    renderer = vtkRenderer()
    renderer.AddActor(actor)
    renderer.AddActor2D(scalarBar)

    window = vtkRenderWindow()
    window.AddRenderer(renderer)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    interactor_style = vtkInteractorStyleTrackballCamera() 
    interactor.SetInteractorStyle(interactor_style)

    window.SetSize((800,600))
    window.Render()
    interactor.Start()

    render_window = interactor.GetRenderWindow()
    render_window.Finalize()


def closestBoundaryId(polydata, objectivePointId, boundary=None, polygons=None, adjMatrix=None):
    """Given a vertex ID, calculate the point ID in any of the boundaries of the
    mesh that is closest to it.

    Returns:
        closesPoint (int): ID of the vertex in the polydata that is part of any boundary, which is closest to objectivePointId.

    Args:
        polydata (vtkPolyData): vtkPolyData object with information of the mesh.
        objectivePointId (int): The point for which the closest boundary point attempts to be calculated.
        boundary (list): List of connected components (boundaries) in the shape of lists.
        polygons (numpy.ndarray): information on the identifiers of every point in every triangle of the mesh.
        adjMatrix (scipy.sparse.csr.csr_matrix): adjacency matrix.
    """

    closestPointIndex       = None
    boundaryVector          = []
    boundaryNumber          = []
    totalBoundaryPoints     = 0

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




def outputLocation(input_path, output_path=None, folder_name=None):
    """Automated localizator for output location given an input path (where the
    input file is). If the output path cannot be accessed or no file can be
    created in the output folder, a location relative to the input location will
    be used. If no output path is provided, a location relative to the input 
    location will be used. If the output path is correctly established, it is
    maintained.

    Returns:
        path (int): Final output path.

    Args:
        input_path  (str): Path of the input file.
        output_path (str): Initial output path.
        folder_name (str): Folder to be used as container for the outputs.
    """

    path                    = None

    if output_path is input_path:
        print(" *  Output path provided coincides with input path.\n"+
              "    Overwriting the input file is not permitted.\n"+
              "    Writing in the default location...\n")

    if output_path is None:
        output_path         = input_path

    if folder_name is None:
        folder_name         = 'Results'

    if output_path is input_path:
        print(" *  Writing to default location: ")

        directory, filename = split(input_path)
        filename, extension = splitext(filename)

        if isdir(join(directory, folder_name)):
            path            = join(directory, folder_name, 
                                   str(filename + '_' + folder_name + extension))
        else:
            mkdir(join(directory, folder_name))

            if isdir(join(directory, folder_name)):
                path        = join(directory, folder_name, 
                                   str(filename + '_' + folder_name + extension))
            else:
                path        = join(directory, str(filename + '_' + folder_name + extension))

        print("    " + path + "\n")

    else:
        directory, filename = split(output_path)

        if not isdir(directory):
            raise RuntimeError("Folder does not exist")

        if splitext(output_path)[1] is '':
            output_path  = output_path + ".vtk"

        path                    = output_path

    return path


def vtkWriterSpanishLocale(path):
    """Solves problems with Spanish locale when writing the output files. In 
    Spanish operating systems, by default, the decimals of a real number are
    specified with commas (,) instead of points (.), rendering the output file
    unreadable for some software that uses '.vtk' files as their input format.

    Args:
        path (str): path in which the substitution of (,) for (.) will take place.
    """

    system("perl -pi -e 's/,/./g' %s " % path)



def vtkInterpolator(QCMHighRes, QCMLowRes, n_neighbors=3, radius=1.0, 
                    algorithm='auto', leaf_size=30, metric='minkowski', 
                    p=5, metric_params=None, n_jobs=1):
    """Produces an interpolation from a low-resolution mesh to a high-resolution
    mesh. The scalar points of the vectors are interpolated, whereas the 3D
    structure of the high-resolution mesh is maintained.

    The interpolation uses K-Nearest Neighbors (from sklearn) using the common
    QCM reference system to identify the closest points.

    Return:
        newPolyData (vtkPolyData): Interpolated vtkPolyData

    Args:
        QCMHighRes (vtkPolyData): high resolution QCM result of PyQCM.endo
        QCMLowRes (vtkPolyData): low resolution QCM result of PyQCM.endo
        n_neighbors, radius, algorithm, leaf_size, ...
        ... metric, p, metric_params, n_jobs: Default sklearn.NearestNeighbors parameters

    """

    # ¿Se podra hacer igual que con TPS?
    kNN = NearestNeighbors(n_neighbors=3, radius=1.0, algorithm='auto', 
                           leaf_size=30, metric='minkowski', p=5, 
                           metric_params=None, n_jobs=1)
    kNN.fit(QCMLowRes.homeomorphism.T, QCMHighRes.homeomorphism.T)
    dist, indices   = kNN.kneighbors(QCMHighRes.homeomorphism.T)

    # The new polydata will have the mesh of the high resolution polydata
    newPolyData     = QCMHighRes.polydata
    newDataArrays   = []

    for i in range(0, QCMLowRes.polydata.GetPointData().GetNumberOfArrays()):
        newDataArrays.append(vtkFloatArray())
        newDataArrays[i].SetName(QCMLowRes.polydata.GetPointData().GetArray(i).GetName())

    scalars_lr      = zeros((QCMLowRes.polydata.GetPointData().GetNumberOfArrays(), QCMLowRes.polydata.GetNumberOfPoints()))
    BIPS            = zeros((QCMLowRes.polydata.GetPointData().GetNumberOfArrays(), QCMHighRes.polydata.GetNumberOfPoints()))

    # Store the polydata scalars into a numpy array
    for i in range(0, QCMLowRes.polydata.GetPointData().GetNumberOfArrays()):
        for j in range(0, QCMLowRes.polydata.GetNumberOfPoints()):
            scalars_lr[i,j] = QCMLowRes.polydata.GetPointData().GetArray(i).GetTuple1(j)

    # If a zero value is found, replace it with the lowest possible number to
    # avoid numerical errors
    zeroIndex               = where(dist == 0.)
    if zeroIndex[0].size > 0:
        for j in range(0, zeroIndex[0].size):
            dist[zeroIndex[0][j], zeroIndex[1][j]] = finfo(dist.dtype).eps

    # Calculate the new scalar values for the low resolution QCM
    for j in range(0, QCMLowRes.polydata.GetPointData().GetNumberOfArrays()):
        for i in range(0, QCMHighRes.polydata.GetNumberOfPoints()):
            dt                  = (1/dist[i,0]) + (1/dist[i,1]) + (1/dist[i,2])
            BIPS[j, i]          = ((1/dist[i,:])*scalars_lr[j, indices[i,:]]).sum()/dt

            newDataArrays[j].InsertNextValue(BIPS[j, i])

    for dataArray in newDataArrays:
        newPolyData.GetPointData().AddArray(dataArray)

    return newPolyData


