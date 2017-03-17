"""
    Copyright (C) 2017 - Universitat Pompeu Fabra
    Author - Guillermo Jimenez-Perez  <guillermo.jim.per@gmail.com>

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

from os import system;
from os import mkdir;

from os.path import isfile;
from os.path import isdir;
from os.path import split;
from os.path import splitext;
from os.path import join;

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
from scipy import dot;
from scipy import cross;

from scipy.sparse import csr_matrix;
from scipy.sparse import spdiags;
from scipy.sparse import find;

from scipy.sparse.linalg import spsolve; 

from mvpoly.rbf import RBFThinPlateSpline;

from vtk import vtkPolyData;
from vtk import vtkPolyDataReader;
from vtk import vtkPolyDataWriter;
from vtk import vtkPoints;
from vtk import vtkIdList;

class VentricularImage(object):
    """ DOCSTRING """

    __imageType                 = None;
    __path                      = None;
    __originalPolyData          = None;
    __QCMPolyData               = None;
    __pointData                 = None;
    __polygonData               = None;
    __scalarData                = None;
    __normalData                = None;
    __nPoints                   = None;
    __nPolygons                 = None;
    __septum                    = None;
    __apex                      = None;
    __laplacianMatrix           = None;
    __boundary                  = None;
    __output                    = None;

    def __init__(self, path, septum, apex, imgType=None):
        """ DOCSTRING """
        if isfile(path):
            self.__path         = path;
        else:
            raise RuntimeError("File does not exist.");
        if imgType is not None:
            if  type(imgType) is type(''):
                self.__imageType = imgType;
            else:
                print("Invalid image type identifier. Ignoring input.") 

        self.__ReadPolyData();
        self.__ReadPointData();
        self.__ReadPolygonData();
        self.__ReadNormalData();
        self.__ReadScalarData();
        self.__septum           = septum;
        self.__apex             = apex;
        self.__nPoints          = self.__pointData.shape[1];
        self.__nPolygons        = self.__polygonData.shape[1];
        self.__CalculateBoundary();
        self.RearrangeBoundary();
        self.__LaplacianMatrix();
        self.__CalculateLinearTransformation();
        self.__CalculateThinPlateSplines();
        self.__WritePolyData();

    def GetPolyData(self):
        return self.__originalPolyData;

    def GetBEP(self):
        return self.__QCMPolyData;

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

    def __ReadPolyData(self):
        reader                  = vtkPolyDataReader();
        reader.SetFileName(self.__path);
        reader.Update();

        polyData                = reader.GetOutput();
        polyData.BuildLinks();

        self.__originalPolyData = polyData;

    def __WritePolyData(self):
        if ((self.__output is not None)
            and (self.__scalarData is not None)
            and (self.__pointData is not None)
            and (self.__polygonData is not None)):

            path                = None;

            directory, filename = split(self.__path);
            filename, extension = splitext(filename);

            newPolyData         = vtkPolyData();
            newPointData        = vtkPoints();
            writer              = vtkPolyDataWriter();

            if isdir(join(directory, 'BEP')):
                path            = join(directory, 'BEP', str(filename + '_BEP' + extension));
                writer.SetFileName(path);
            else:
                mkdir(join(directory, 'BEP'));

                if isdir(join(directory, 'BEP')):
                    path        = join(directory, 'BEP', str(filename + '_BEP' + extension));
                    writer.SetFileName(path);
                else:
                    path        = join(directory, str(filename + '_BEP' + extension));
                    writer.SetFileName(path);

            for i in xrange(self.__nPoints):
                newPointData.InsertPoint(i, (self.__output[0, i], self.__output[1, i], 0.0));

            newPolyData.SetPoints(newPointData);
            newPolyData.SetPolys(self.__originalPolyData.GetPolys());
            if self.__originalPolyData.GetPointData().GetScalars() is None:
                newPolyData.GetPointData().SetScalars(
                    self.__originalPolyData.GetPointData().GetArray(0));
            else:
                newPolyData.GetPointData().SetScalars(
                    self.__originalPolyData.GetPointData().GetScalars());

            writer.SetInputData(newPolyData);
            writer.Write();

            self.__QCMPolyData  = newPolyData;

            system("perl -pi -e 's/,/./g' %s " % path);

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
                polygons        = zeros((rows,cols), dtype=int);
            
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
                    points      = zeros((rows,cols));
                
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
                    normals     = zeros((rows,cols));
                
                normals[0,i]    = normalTuple[0];
                normals[1,i]    = normalTuple[1];
                normals[2,i]    = normalTuple[2];

        self.__normalData       = normals;

    def __ReadScalarData(self):
        rows                    = None;
        cols                    = None;
        scalars                 = None;

        scalarVector            = self.__originalPolyData.GetPointData().GetScalars();

        if scalarVector is None:
            scalarVector        = self.__originalPolyData.GetPointData().GetArray(0);

        if scalarVector:
            for i in xrange(scalarVector.GetNumberOfTuples()):
                scalarTuple     = scalarVector.GetTuple(i);

                if scalars is None:
                    rows        = len(scalarTuple);
                    cols        = scalarVector.GetNumberOfTuples();
                    scalars     = zeros((rows,cols));
                
                for j in xrange(len(scalarTuple)):
                    scalars[j,i] = scalarTuple[j];
        else:
            print("The input file does not have any associated scalar data.")

        self.__scalarData       = scalars;

    def __LaplacianMatrix(self):
        numDims                 = self.__polygonData.shape[0];
        numPoints               = self.__pointData.shape[1];
        numPolygons             = self.__polygonData.shape[1];
        boundary                = self.__boundary;
        boundaryConstrain       = zeros((2,numPoints));

        sparseMatrix            = csr_matrix((numPoints, numPoints));

        for i in range(0, numDims):
            i1                  = (i + 0)%3;
            i2                  = (i + 1)%3;
            i3                  = (i + 2)%3;

            distP2P1            = self.__pointData[:, self.__polygonData[i2, :]] - self.__pointData[:, self.__polygonData[i1, :]];
            distP3P1            = self.__pointData[:, self.__polygonData[i3, :]] - self.__pointData[:, self.__polygonData[i1, :]];

            distP2P1            = distP2P1 / repmat(sqrt((distP2P1**2).sum(0)), 3, 1);
            distP3P1            = distP3P1 / repmat(sqrt((distP3P1**2).sum(0)), 3, 1);

            angles              = arccos((distP2P1 * distP3P1).sum(0));

            iterData1           = csr_matrix((1/tan(angles), 
                                                    (self.__polygonData[i2,:], 
                                                     self.__polygonData[i3,:])), 
                                                    shape=(numPoints, numPoints));

            iterData2           = csr_matrix((1/tan(angles), (self.__polygonData[i3,:], self.__polygonData[i2,:])), shape=(numPoints, numPoints));

            sparseMatrix        = sparseMatrix + iterData1 + iterData2;

        diagonal                = sparseMatrix.sum(0);
        diagonalSparse          = spdiags(diagonal, 0, numPoints, numPoints);
        self.__laplacianMatrix  = diagonalSparse - sparseMatrix;

    def __CalculateLinearTransformation(self):
        if self.__laplacianMatrix is not None:
            if self.__boundary is not None:
                laplacian       = self.__laplacianMatrix;
                (nzi, nzj)      = find(laplacian)[0:2];

                for point in self.__boundary:
                    positions   = where(nzi==point)[0];

                    laplacian[nzi[positions], nzj[positions]] = 0;

                    laplacian[point, point] = 1;

                Z = self.GetWithinBoundarySinCos();

                boundaryConstrain = zeros((2, self.__nPoints));
                boundaryConstrain[:, self.__boundary] = Z;

                self.__output   = spsolve(laplacian, boundaryConstrain.transpose()).transpose();

    def __CalculateThinPlateSplines(self):
        if self.__output is not None:
            if self.__apex is not None:
                boundaryPoints  = self.__output[:,self.__boundary];
                source          = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));
                destination     = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));

                source[:, 0:source.shape[1] - 1]        = boundaryPoints;
                source[:, source.shape[1] - 1]          = self.__output[:, self.__apex];

                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;
                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;

                x = source[0,:];
                y = source[1,:];
                d = destination[0,:] + 1j*destination[1,:];

                thinPlateInterpolation = RBFThinPlateSpline(x,y,d);

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
            cellPointIdList     = vtkIdList();
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

                    edgeIdList  = vtkIdList()
                    edgeIdList.InsertNextId(cellEdges[i][0]);
                    edgeIdList.InsertNextId(cellEdges[i][1]);

                    singleCellEdgeNeighborIds = vtkIdList();

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
            neighboringCells    = vtkIdList();

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

                edgeNeighbors1  = vtkIdList();
                edgeNeighbors2  = vtkIdList();

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

        boundary    = asarray(boundary, dtype=int);

        center      = mean(self.__pointData[:,boundary], axis=1);
        vector1     = asarray(self.__pointData[:,boundary[0]] - center);
        vector2     = asarray(self.__pointData[:,boundary[1]] - center);
        vectorNormal= cross(vector1, vector2);
        vectorApex  = self.__pointData[:, self.__apex] - center;

        if len(center.shape) is not 1:
            if center.shape[0] is not 3:
                raise Exception("Something went wrong. Probably forgot to transpose this. Contact maintainer.");

        if dot(vectorApex, vectorNormal) < 0:
            boundary         = flip(boundary, 0);
            boundary         = roll(boundary, 1);

        self.__boundary = boundary;

    def FlipBoundary(self):
        self.__boundary         = flip(self.__boundary, 0);
        self.__boundary         = roll(self.__boundary, 1);

    def GetWithinBoundaryDistances(self):
        boundaryNext            = roll(self.__boundary, -1);
        boundaryNextPoints      = self.__pointData[:, boundaryNext];

        distanceToNext          = boundaryNextPoints - self.GetBoundaryPoints();

        return sqrt((distanceToNext**2).sum(0));

    def GetPerimeter(self):
        return self.GetWithinBoundaryDistances().sum();

    def GetWithinBoundaryDistancesAsFraction(self):
        euclideanNorm           = self.GetWithinBoundaryDistances();
        perimeter               = euclideanNorm.sum();

        return euclideanNorm/perimeter;

    def GetWithinBoundaryAngles(self):
        circleLength            = 2*pi;
        fraction                = self.GetWithinBoundaryDistancesAsFraction();

        angles                  = cumsum(circleLength*fraction);
        angles                  = roll(angles, 1);
        angles[0]               = 0;

        return angles;

    def GetWithinBoundarySinCos(self):
        angles                  = self.GetWithinBoundaryAngles();
        Z                       = zeros((2, angles.size));
        Z[0,:]                  = cos(angles);
        Z[1,:]                  = sin(angles);

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
            closestPointIndex   = where(self.__boundary==septalIndex);

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary     = roll(self.__boundary, -closestPointIndex);
        else:
            try:
                septalPoint     = self.__pointData[:, septalIndex];
            except:
                raise Exception("Septal point provided out of data bounds; the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.");

            if len(self.__boundary.shape) == 1:
                septalPoint     = repmat(septalPoint,
                                    self.__boundary.size, 1);
                septalPoint     = septalPoint .transpose();
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.");

            distanceToObjectivePoint    = (self.__pointData[:, self.__boundary] - septalPoint);
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0));
            closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min());
            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary             = roll(self.__boundary, 
                                          -closestPointIndex);

    # def SetInputConnection():













# # import scipy, time, os, numpy, VentricularImage;

# import time, os, VentricularImage;

# septum_MRI = 201479 - 1;
# apex_MRI = 37963 - 1;
# path_MRI = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

# apex_EAM = 599 - 1;
# septum_EAM = 1389 - 1;
# path_EAM = os.path.join("/home/guille/BitBucket/qcm/data/pat1/EAM", "pat1_EAM_endo_smooth.vtk");

# start = time.time(); reload(VentricularImage); MRI = VentricularImage.VentricularImage(path_MRI, septum_MRI, apex_MRI); print(time.time() - start);
# start = time.time(); reload(VentricularImage); EAM = VentricularImage.VentricularImage(path_EAM, septum_EAM, apex_EAM); print(time.time() - start);
















# class MouseInteractorHighLightActor(vtk.vtkInteractorStyleTrackballCamera):
 
#     def __init__(self,parent=None):
#         self.AddObserver("LeftButtonPressEvent",self.leftButtonPressEvent)
 
#         self.LastPickedActor = None
#         self.LastPickedProperty = vtk.vtkProperty()
 
#     def leftButtonPressEvent(self,obj,event):
#         clickPos = self.GetInteractor().GetEventPosition()
 
#         picker = vtk.vtkPropPicker()
#         picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
 
#         # get the new
#         self.NewPickedActor = picker.GetActor()
 
#         # If something was selected
#         if self.NewPickedActor:
#             # If we picked something before, reset its property
#             if self.LastPickedActor:
#                 self.LastPickedActor.GetProperty().DeepCopy(self.LastPickedProperty)
 
 
#             # Save the property of the picked actor so that we can
#             # restore it next time
#             self.LastPickedProperty.DeepCopy(self.NewPickedActor.GetProperty())
#             # Highlight the picked actor by changing its properties
#             self.NewPickedActor.GetProperty().SetColor(1.0, 0.0, 0.0)
#             self.NewPickedActor.GetProperty().SetDiffuse(1.0)
#             self.NewPickedActor.GetProperty().SetSpecular(0.0)
 
#             # save the last picked actor
#             self.LastPickedActor = self.NewPickedActor
 
#         self.OnLeftButtonDown()
#         return




# class MouseInteractor(vtk.vtkInteractorStyleTrackballCamera):
#     __mapper                    = None;
#     __actor                     = None;

#     def __init__(self):
#         self.__mapper           = vtk.vtkDataSetMapper();
#         self.__actor            = vtk.vtkActor();

#     def OnLeftButtonDown(self):
#         position                = self.GetInteractor().GetEventPosition();
#         picker                  = vtk.vtkCellPicker();
#         picker.SetTolerance(0.0005);



#     def GetMapper(self):
#         return self.__mapper;

#     def GetActor(self):
#         return self.__actor;
         


# >>> A = MRI.GetPolyData();
# >>> 
# >>> 
# >>> mapper = vtk.vtkPolyDataMapper();
# >>> 
# >>> 
# >>> mapper.SetInputData(A);
# >>> 
# >>> actor = vtk.vtkActor();
# >>> 
# >>> actor.SetMapper(mapper);
# >>> 
# >>> 
# >>> trackball = vtk.vtkInteractorStyleTrackballCamera();
# >>> 
# >>> renderer = vtk.vtkRenderer();
# >>> 
# >>> 
# >>> rendererWindow = vtk.vtkRenderWindow();
# >>> 
# >>> 
# >>> rendererWindowInteractor = vtk.vtkRenderWindowInteractor();
# >>> 
# >>> 
# >>> rendererWindowInteractor.SetRenderWindow(rendererWindow);
# >>> 
# >>> 
# >>> trackball.SetDefaultRenderer(renderer);
# >>> 
# >>> rendererWindowInteractor.SetInteractorStyle(trackball);
# >>> 
# >>> 
# >>> renderer.AddActor(actor);
# >>> renderer.SetBackground( 0,0,1 );
# >>> 
# >>> rendererWindow.Render();
# >>> rendererWindowInteractor.Initialize()
# >>> rendererWindowInteractor.Start()



