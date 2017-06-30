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

from scipy.sparse.linalg import spsolve
from mvpoly.rbf import RBFThinPlateSpline







class PyQCM(BaseImage):
    __path                      = None

    __polydata                  = None
    __points                    = None
    __polygons                  = None

    __scalars                   = None
    __normals                   = None

    __npoints                   = None
    __npolygons                 = None
    __ndim                      = None
    __nscalars                  = None
    __nedges_mesh               = None
    __scalars_names             = None

    __septum                    = None
    __apex                      = None
    __laplacian                 = None
    __boundary                  = None
    __QCM_polydata              = None
    __QCM_points                = None
    __QCM_path                  = None


    def __init__(self, path, septum, apex, output_path=None):
        if isfile(path):
            self.__path         = path
        else:
            raise RuntimeError("File does not exist")

        self.__read_polydata()
        self.__read_points()
        self.__read_polygons()
        self.__read_normals()
        self.__read_scalars()
        self.__septum           = septum
        self.__apex             = apex

        if output_path is None:
            self.__QCM_path     = self.path
        else:
            if output_path is self.path:
                print(" *  Output path provided coincides with input path.\n"+
                      "    Overwriting the input file is not permitted.\n"+
                      "    Writing in the default location...\n")
                self.__QCM_path = self.path
            else:
                self.__QCM_path = output_path

        self.__calc_boundary()
        self.__rearrange()
        self.__calc_laplacian()
        self.__calc_QCM_points()
        self.__write_QCM_polydata()

    @property
    def path(self):
        return self.__path

    @property
    def QCM_path(self):
        return self.__QCM_path

    @property
    def polydata(self):
        """Testing docstring of attribute"""
        return self.__polydata

    @property
    def ndim(self):
        return self.__ndim

    @property
    def nedges_mesh(self):
        return self.__nedges_mesh

    @property
    def points(self):
        return self.__points

    @property
    def npoints(self):
        return self.__npoints

    @property
    def polygons(self):
        return self.__polygons

    @property
    def npolygons(self):
        return self.__npolygons

    @property
    def scalars(self):
        return self.__scalars

    @property
    def scalars_names(self):
        return self.__scalars_names

    @property
    def normals(self):
        return self.__normals

    @QCM_path.setter
    def QCM_path(self, output_path):
        if output_path is self.path:
            print(" *  Warning! Overwriting the input file is not permitted.\n"
                  "    Aborting...\n")
            return
        else:
            if self.QCM_path == self.path:
                print(" *  Warning! The file written to the default location will *not*\n"
                      "    be deleted\n")
            else:
                print(" *  Warning! The file written to the previous working location will \n"
                      "    *not* be deleted\n")

        self.__QCM_path         = output_path
        self.__write_QCM_polydata()


    @property
    def septum(self):
        return self.__septum

    @septum.setter
    def septum(self, septum):
        if septum >= self.npoints:
            raise RuntimeError("Septal point provided is out of bounds")

        self.rearrange_boundary(septum)

    @property
    def apex(self):
        return self.__apex

    @apex.setter
    def apex(self, apex):
        if apex >= self.npoints:
            raise RuntimeError("Apical point provided is out of bounds")

        self.__apex             = apex
        self.__calc_QCM_points()
        self.__write_QCM_polydata()

    @property
    def laplacian(self):
        return self.__laplacian

    @property
    def boundary(self):
        return self.__boundary

    @property
    def boundary_points(self):
        return self.points[:, self.boundary]

    @property
    def QCM_points(self):
        return self.__QCM_points

    @property
    def QCM_polydata(self):
        return self.__QCM_polydata


    def __read_polydata(self):
        reader                  = vtkPolyDataReader()
        reader.SetFileName(self.path)
        reader.Update()

        polydata                = reader.GetOutput()
        polydata.BuildLinks()

        self.__npoints          = polydata.GetNumberOfPoints()
        self.__npolygons        = polydata.GetNumberOfPolys()
        self.__ndim             = polydata.GetPoints().GetData().GetNumberOfComponents()
        self.__polydata         = polydata


    def __calc_laplacian(self):
for j in range(10):
    start = time.time()

    numPoints    = polydata.GetNumberOfPoints()
    numPolygons  = polydata.GetNumberOfPolys()
    sparseMatrix = scipy.sparse.csr_matrix((numPoints, numPoints))

    try:
        rows_polys      = polydata.GetCell(0).GetPointIds().GetNumberOfIds()
        cols_polys      = polydata.GetNumberOfCells()
        polygons        = numpy.zeros((rows_polys,cols_polys), dtype=int)

        rows_points     = len(polydata.GetPoint(0))
        cols_points     = polydata.GetPoints().GetNumberOfPoints()
        points          = numpy.zeros((rows_points,cols_points))

        numDims         = rows_points
    except:
        raise Exception('The polydata provided is empty')

    # Exporting VTK triangles to numpy for a more efficient manipulation
    for i in xrange(polydata.GetNumberOfCells()):
        triangle            = polydata.GetCell(i)
        pointIds            = triangle.GetPointIds()

        polygons[0,i]       = pointIds.GetId(0)
        polygons[1,i]       = pointIds.GetId(1)
        polygons[2,i]       = pointIds.GetId(2)


    # Exporting VTK points to numpy for a more efficient manipulation
    pointVector             = polydata.GetPoints()

    if pointVector:
        for i in range(0, pointVector.GetNumberOfPoints()):
            point_tuple     = pointVector.GetPoint(i)

            if points is None:
                rows        = len(point_tuple)
                cols        = pointVector.GetNumberOfPoints()
                points      = numpy.zeros((rows,cols))

            points[0,i]     = point_tuple[0]
            points[1,i]     = point_tuple[1]
            points[2,i]     = point_tuple[2]

    # Calculation of Laplacian
    for i in range(0, numDims):
        i1                  = (i + 0)%3
        i2                  = (i + 1)%3
        i3                  = (i + 2)%3

        vectP2P1            = points[:, polygons[i2, :]] - points[:, polygons[i1, :]]
        vectP3P1            = points[:, polygons[i3, :]] - points[:, polygons[i1, :]]

        vectP2P1            = vectP2P1 / numpy.matlib.repmat(numpy.sqrt((vectP2P1**2).sum(0)), numDims, 1)
        vectP3P1            = vectP3P1 / numpy.matlib.repmat(numpy.sqrt((vectP3P1**2).sum(0)), numDims, 1)

        angles              = scipy.arccos((vectP2P1 * vectP3P1).sum(0))

        iterData1           = scipy.sparse.csr_matrix((1/scipy.tan(angles), 
                                                      (polygons[i2,:], 
                                                       polygons[i3,:])), 
                                                       shape=(numPoints, numPoints))

        iterData2           = scipy.sparse.csr_matrix((1/scipy.tan(angles), (polygons[i3,:], polygons[i2,:])), shape=(numPoints, numPoints))

        sparseMatrix        = sparseMatrix + iterData1 + iterData2

    diagonal                = sparseMatrix.sum(0)
    diagonalSparse          = spdiags(diagonal, 0, numPoints, numPoints)
    self.laplacian          = diagonalSparse - sparseMatrix

    
    def __calc_QCM_points(self):
        if self.laplacian is not None:
            if self.boundary is not None:
                (nzi, nzj)      = find(self.laplacian)[0:2]

                for point in self.boundary:
                    positions   = where(nzi==point)[0]

                    self.laplacian[nzi[positions], nzj[positions]] = 0

                    self.laplacian[point, point] = 1

                angles                  = self.__calc_boundary_node_angles()
                Z                       = zeros((2, angles.size))
                Z[0,:]                  = cos(angles)
                Z[1,:]                  = sin(angles)

                boundaryConstrain = zeros((2, self.npoints))
                boundaryConstrain[:, self.boundary] = Z

                self.__QCM_points = spsolve(self.laplacian, boundaryConstrain.transpose()).transpose()

                self.__calc_thin_plate_splines()

    def __calc_thin_plate_splines(self):
        if self.QCM_points is not None:
            if self.apex is not None:
                boundaryPoints  = self.QCM_points[:,self.boundary]
                source          = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1))
                destination     = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1))

                source[:, 0:source.shape[1] - 1]        = boundaryPoints
                source[:, source.shape[1] - 1]          = self.QCM_points[:, self.apex]

                destination[:, 0:source.shape[1] - 1]   = boundaryPoints
                destination[:, 0:source.shape[1] - 1]   = boundaryPoints

                x = source[0,:]
                y = source[1,:]
                d = destination[0,:] + 1j*destination[1,:]

                thinPlateInterpolation = RBFThinPlateSpline(x,y,d)

                result = thinPlateInterpolation(self.QCM_points[0,:], self.QCM_points[1,:])

                self.__QCM_points[0,:] = result.real
                self.__QCM_points[1,:] = result.imag

    def __calc_boundary(self):
        startingPoint           = None
        currentPoint            = None
        foundBoundary           = False
        cellId                  = None
        boundary                = []
        visitedEdges            = []
        visitedBoundaryEdges    = []

        for cellId in xrange(self.polydata.GetNumberOfCells()):
            cellPointIdList     = vtkIdList()
            cellEdges           = []

            self.polydata.GetCellPoints(cellId, cellPointIdList)

            cellEdges           = [[cellPointIdList.GetId(0), 
                                    cellPointIdList.GetId(1)], 
                                   [cellPointIdList.GetId(1), 
                                    cellPointIdList.GetId(2)], 
                                   [cellPointIdList.GetId(2), 
                                    cellPointIdList.GetId(0)]]

            for i in xrange(len(cellEdges)):
                if (cellEdges[i] in visitedEdges) == False:
                    visitedEdges.append(cellEdges[i])

                    edgeIdList  = vtkIdList()
                    edgeIdList.InsertNextId(cellEdges[i][0])
                    edgeIdList.InsertNextId(cellEdges[i][1])

                    singleCellEdgeNeighborIds = vtkIdList()

                    self.polydata.GetCellEdgeNeighbors(cellId, cellEdges[i][0], cellEdges[i][1], singleCellEdgeNeighborIds)

                    if singleCellEdgeNeighborIds.GetNumberOfIds() == 0:
                        foundBoundary   = True

                        startingPoint   = cellEdges[i][0]
                        currentPoint    = cellEdges[i][1]

                        boundary.append(cellEdges[i][0])
                        boundary.append(cellEdges[i][1])

                        visitedBoundaryEdges.append([currentPoint,startingPoint])
                        visitedBoundaryEdges.append([startingPoint,currentPoint])

            if foundBoundary == True:
                break

        if foundBoundary == False:
            raise Exception("The mesh provided has no boundary not possible to do Quasi-Conformal Mapping on this dataset.")

        while currentPoint != startingPoint:
            neighboringCells    = vtkIdList()

            self.polydata.GetPointCells(currentPoint, neighboringCells)

            for i in xrange(neighboringCells.GetNumberOfIds()):
                cell = neighboringCells.GetId(i)
                triangle = self.polydata.GetCell(cell)

                for j in xrange(triangle.GetNumberOfPoints()):
                    if triangle.GetPointId(j) == currentPoint:
                        j1      = (j + 1) % 3
                        j2      = (j + 2) % 3

                        edge1   = [triangle.GetPointId(j),
                             triangle.GetPointId(j1)]
                        edge2   = [triangle.GetPointId(j),
                             triangle.GetPointId(j2)]

                edgeNeighbors1  = vtkIdList()
                edgeNeighbors2  = vtkIdList()

                self.polydata.GetCellEdgeNeighbors(cell, edge1[0], edge1[1], edgeNeighbors1)

                self.polydata.GetCellEdgeNeighbors(cell, edge2[0], edge2[1], edgeNeighbors2)

                if edgeNeighbors1.GetNumberOfIds() == 0:
                    if ([edge1[1], edge1[0]] in visitedBoundaryEdges) == False:
                        if (edge1[1] in boundary) == False:
                            boundary.append(edge1[1])
                        visitedBoundaryEdges.append([edge1[0], edge1[1]])
                        visitedBoundaryEdges.append([edge1[1], edge1[0]])
                        currentPoint = edge1[1]
                        break

                if edgeNeighbors2.GetNumberOfIds() == 0:
                    if ([edge2[1], edge2[0]] in visitedBoundaryEdges) == False:
                        if (edge2[1] in boundary) == False:
                            boundary.append(edge2[1])
                        visitedBoundaryEdges.append([edge2[0], edge2[1]])
                        visitedBoundaryEdges.append([edge2[1], edge2[0]])
                        currentPoint = edge2[1]
                        break

        boundary    = asarray(boundary, dtype=int)

        center      = mean(self.points[:,boundary], axis=1)
        vector1     = asarray(self.points[:,boundary[0]] - center)
        vector2     = asarray(self.points[:,boundary[1]] - center)
        vectorNormal= cross(vector1, vector2)
        vectorApex  = self.points[:, self.apex] - center

        if len(center.shape) is not 1:
            if center.shape[0] is not 3:
                raise Exception("Something went wrong. Probably forgot to transpose this. Contact maintainer.")

        if dot(vectorApex, vectorNormal) < 0:
            boundary            = flipud(boundary)
            boundary            = roll(boundary, 1)

        self.__boundary         = boundary

    def flip_boundary(self):
        self.__boundary         = flipud(self.boundary)
        self.__boundary         = roll(self.boundary, 1)

    def get_boundary_node_distances(self):
        boundaryNext            = roll(self.boundary, -1)
        boundaryNextPoints      = self.points[:, boundaryNext]

        distanceToNext          = boundaryNextPoints - self.boundary_points

        return sqrt((distanceToNext**2).sum(0))

    def get_boundary_perimeter(self):
        return self.get_boundary_node_distances().sum()

    def get_boundary_node_distances_fraction(self):
        euclideanNorm           = self.get_boundary_node_distances()
        perimeter               = euclideanNorm.sum()

        return euclideanNorm/perimeter

    def __calc_boundary_node_angles(self):
        circleLength            = 2*pi
        fraction                = self.get_boundary_node_distances_fraction()

        angles                  = cumsum(circleLength*fraction)
        angles                  = roll(angles, 1)
        angles[0]               = 0

        return angles

    def __rearrange(self, objectivePoint=None):
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



    def rearrange_boundary(self, objectivePoint):
        """Rearranges the boundary aroung a new point identifier
        """
        old_septum              = self.septum
        septalIndex             = None
        closestPoint            = None

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

            # self.__boundary = roll(self.boundary, -septalIndex)
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
        vector2 = self.QCM_points[:, septalIndex] - center

        angle   = arccos(dot(vector1, vector2)/(norm(vector1)*norm(vector2)))

        # If the y coordinate of the vector w.r.t. the rotation will take place
        # is negative, the rotation must be done counterclock-wise
        if vector2[1] > 0:
            angle = -angle

        rotation_matrix                 = asarray([[cos(angle), -sin(angle)],
                                                   [sin(angle), cos(angle)]])

        self.__septum                   = septalIndex
        self.__QCM_points               = rotation_matrix.dot(self.QCM_points)

        self.__write_QCM_polydata()

    def __write_QCM_polydata(self):
        if ((self.QCM_points is not None)
            and (self.points is not None)
            and (self.scalars is not None)
            and (self.polygons is not None)):

            newPolyData         = vtkPolyData()
            newPointData        = vtkPoints()
            writer              = vtkPolyDataWriter()

            if self.QCM_path is self.path:
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
                if splitext(self.QCM_path)[1] is '':
                    self.__QCM_path = self.QCM_path + ".vtk"

                path                = self.QCM_path


            writer.SetFileName(path)

            for i in xrange(self.npoints):
                newPointData.InsertPoint(i, (self.QCM_points[0, i], self.QCM_points[1, i], 0.0))

            newPolyData.SetPoints(newPointData)
            newPolyData.SetPolys(self.polydata.GetPolys())
            if self.polydata.GetPointData().GetScalars() is None:
                newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetArray(0))
            else:
                newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetScalars())

            writer.SetInputData(newPolyData)
            writer.Write()

            self.__QCM_polydata  = newPolyData

            system("perl -pi -e 's/,/./g' %s " % path)

        else:
            raise RuntimeError("Information provided insufficient")

    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path + "'.\n"
        s = s + "Number of dimensions: " + str(self.ndim) + "\n"
        s = s + "Number of points: " + str(self.npoints) + "\n"
        s = s + "Number of polygons: " + str(self.npolygons) + "\n"
        s = s + "Number of edges of the polygons: " + str(self.nedges_mesh) + "\n"
        s = s + "Scalar information: " + str(self.scalars_names)
        s = s + "Output file location: " + str(self.QCM_path)
        return s
















