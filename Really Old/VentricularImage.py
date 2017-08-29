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

from AbstractImage import BaseImage;

from os import system;
from os import mkdir;

# from os.path import isfile;
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
from scipy import flipud;
from scipy import roll;
from scipy import dot;
from scipy import cross;

from scipy.linalg import norm;

from scipy.sparse import csr_matrix;
from scipy.sparse import spdiags;
from scipy.sparse import find;

from scipy.sparse.linalg import spsolve;

from mvpoly.rbf import RBFThinPlateSpline;

from vtk import vtkPolyData;
from vtk import vtkPolyDataWriter;
from vtk import vtkPoints;
from vtk import vtkIdList;

class VentricularQCM(BaseImage):
    """
    Data extraction from a VTK file containing information from a ventricle
    chamber. A quasi-conformal mapping of the mesh (f: R2->R3) will be produced,
    taking the information of the scalar fields associated to the mesh.

    This object creates a vtk file containing the information of the quasi
    conformal mapping on either a specified output location (QCM_output_path)
    or the same location of the input file (adding a "_QCM" suffix).



    Attributes
    ----------

    septum:             int
        identifier of the point that marks the septal point of the chamber

    apex:               int
        identifier of the point that marks the apical point of the chamber

    laplacian:          scipy.sparse.csr.csr_matrix
        laplacian matrix of the connectivity of the input VTK mesh, in the 
        form of a sparse matrix

    boundary:           numpy.ndarray
        array of the identifiers of the points that compose the boundary, 
        ordered by connectivity and following a clockwise orientation

    QCM_points:         numpy.ndarray
        numpy array containing the two-dimensional points resulting from the
        application of the quasi-conformal mapping and a thin-plate splines
        transformation

    QCM_polydata:       vtk.vtkPolyData
        polydata containing the information of the ventricle after applying
        the quasi-conformal mapping from the 3D input VTK file

    QCM_path:           str
        (optional) defaults to the same path as the input file, creating a 
        folder in the input file's folder (named QCM) and, inside of that
        folder, a file with the same name as the input folder followed by 
        "_QCM" and the extension

    <Attributes inherited of BaseImage>
    path:               str
        path to the image file

    ndim:               int
        dimensionality of the VTK image

    npoints:            int
        number of points in the VTK file

    npolygons:          int
        number of polygons in the VTK file

    nedges_mesh:        int
        number of edges of the mesh in the input VTK file. Determines the
        shape of the mesh; whether it is triangular, quadrilateral or of a
        higher order

    nscalars:           int
        number of scalar vectors in the input VTK file

    polydata:           vtkPolyData
        link to the original vtkPolyData stored in the 'path' variable

    points:             numpy.ndarray
        coordinates of the point in a numpy array, in shape (ndim, npoints)

    polygons:           numpy.ndarray
        point identifiers of each of the triangles of the mesh, 
        in shape (nedges_mesh, npolygons). If a point of a mesh cell wants to
        be accessed, the specific point ID has to be passed to the points array:

        >>> # (being ob the BaseImage object)
        >>> ob.nedges_mesh # The mesh is triangular
        3
        >>> ob.polygons[:,35] # Let's check the point IDs of the 36th triangle
        array([286, 715, 69])
        >>> # If we wanted to check the spatial coordinates of the first node
        >>> # of the 36th triangle, that is, the 287th node of the set, it
        >>> # could be accessed by passing the 1st node of the 36th triangle
        >>> # to the points array:
        >>> point_of_interest = ob.points[:, ob.polygons[0, 35]]
        array([  56.53070068, -101.97899628,  179.80900574]) 

    scalars:            numpy.ndarray
        structured numpy array containing the information of the scalars
        associated to each point coordinate, in shape (nscalars, npoints). It
        allows for specific attribute calls. E.g:  if a VTK file with two scalar
        fields is analyzed, the first being called 'scalars' and the second
        'LAT', the following statement is valid:

        >>> # (being ob the BaseImage object)
        >>> ob.scalars['LAT']
        array([ 0.52275603,  0.94302633,  0.81044762, ...,  0.02766091,
                0.18389358,  0.62247573])

    scalars_names:      tuple
        wrapper for 'scalars.dtype.names'. Provides the names of the scalar 
        arrays contained in the input VTK file

    normals:            numpy.ndarray
        numpy array containing the normals of each point, in shape 
        (ndim, npoints)



    Returns
    -------

    output:         VentricularQCM
        VentricularQCM object containing the extracted information of the input
        VTK file



    See Also
    --------

    BaseImage
    VentricularInterpolator



    Example
    -------

    >>> import VentricularImage
    >>> import os
    >>> input_path = os.path("/path/to/image.vtk")
    >>> septum = 1894
    >>> apex = 1098
    >>> output_path = os.path("/desired/path/to/output/file.vtk")
    >>> output_path = os.path("/desired/path/to/output/file") # Also works
    >>> image1 = VentricularImage.VentricularQCM(input_path, septum, apex)
     *  Writing to default location: 
        /home/qcm/data/pat1/EAM/QCM/pat1_EAM_endo_smooth_QCM.vtk
    
    >>> polydata = image1.polydata
    >>> polydata.GetNumberOfPoints()
    489618
    >>> image1.npoints
    489618
    >>> image1.boundary
    array([1388, 1239, 1244, 1404, 1400, 1176, 1363, 1181, 1313, 1186, 1253,
           1183, 1258, 1320, 1105, 1316, 1365, 1015, 1009,  913,  935, 1394,
            907,  805,  799,  793,  787,  881, 1371,  934,  875, 1330, 1282,
           1219, 1012, 1096, 1182, 1177, 1249, 1309, 1291, 1339, 1362, 1344,
           1383, 1403])
    >>> image1.boundary_points[:, 0:4]
    array([[  -5.78809977,   -9.1422596 ,   -6.71489   ,   -5.78809977],
           [ -93.72029877,  -93.72029877,  -97.84989929,  -99.01100159],
           [ 145.13800049,  146.772995  ,  146.772995  ,  146.772995  ]])
    >>> 
    >>> new_septum = 4897
    >>> new_septum in image1.boundary
    False
    >>> image1.septum = new_septum
     *  Provided point not found in the boundary. Selecting 
    closest point available...
    >>> new apex = 181
    >>> image1.apex = new_apex
     *  Writing to default location: 
    /home/qcm/data/pat1/EAM/QCM/pat1_EAM_endo_smooth_QCM.vtk
    >>> image1.QCM_path = input_path
     *  Warning! Overwriting the input file is not permitted.
    Aborting...
    >>> new_output_path = "/another/path/to/output/file" # No need for os.path
    >>> image1.QCM_path = new_output_path
     *  Warning! The file written to the default location will *not*
    be deleted

    """

    __septum                    = None;
    __apex                      = None;
    __laplacian                 = None;
    __boundary                  = None;
    __QCM_polydata              = None;
    __QCM_points                = None;
    __QCM_path                  = None;


    def __init__(self, path, septum, apex, output_path=None):
        """ VentricularImage(path, septum, apex)

        Analyzes a ventricular image in vtk format and creates quasi-conformal 
        mapping. Requires the information of the input path, the IDs of the 
        septal and the apical points and, optionally, the desired output path.

        The object VentricularQCM provides an automatic traduction of a image
        file in vtk format to a quasi-conformal disk image to be further 
        analyzed by other tools. 

        For more information, type help(VentricularImage.VentricularQCM);
        """

        BaseImage.__init__(self, path);

        self.__septum           = septum;
        self.__apex             = apex;
        if output_path is None:
            self.__QCM_path     = self.path;
        else:
            if output_path is self.path:
                print(" *  Output path provided coincides with input path.\n"+
                      "    Overwriting the input file is not permitted.\n"+
                      "    Writing in the default location...\n")
                self.__QCM_path = self.path;
            else:
                self.__QCM_path = output_path;

        self.__calc_boundary();
        self.__rearrange();
        self.__calc_laplacian();
        self.__calc_QCM_points();
        self.__write_QCM_polydata();

    @property
    def QCM_path(self):
        return self.__QCM_path;


    @QCM_path.setter
    def QCM_path(self, output_path):
        if output_path is self.path:
            print(" *  Warning! Overwriting the input file is not permitted.\n"
                  "    Aborting...\n")
            return
        else:
            if self.QCM_path == self.path:
                print(" *  Warning! The file written to the default location will *not*\n"
                      "    be deleted\n");
            else:
                print(" *  Warning! The file written to the previous working location will \n"
                      "    *not* be deleted\n");

        self.__QCM_path         = output_path;
        self.__write_QCM_polydata();


    @property
    def septum(self):
        return self.__septum;

    @septum.setter
    def septum(self, septum):
        if septum >= self.npoints:
            raise RuntimeError("Septal point provided is out of bounds");

        self.rearrange_boundary(septum);

    @property
    def apex(self):
        return self.__apex;

    @apex.setter
    def apex(self, apex):
        if apex >= self.npoints:
            raise RuntimeError("Apical point provided is out of bounds");

        self.__apex             = apex;
        self.__calc_QCM_points();
        self.__write_QCM_polydata();

    @property
    def laplacian(self):
        return self.__laplacian;

    @property
    def boundary(self):
        return self.__boundary;

    @property
    def boundary_points(self):
        return self.points[:, self.boundary];

    @property
    def QCM_points(self):
        return self.__QCM_points;

    @property
    def QCM_polydata(self):
        return self.__QCM_polydata;

    def __write_QCM_polydata(self):
        if ((self.QCM_points is not None)
            and (self.points is not None)
            and (self.scalars is not None)
            and (self.polygons is not None)):

            newPolyData         = vtkPolyData();
            newPointData        = vtkPoints();
            writer              = vtkPolyDataWriter();

            if self.QCM_path is self.path:
                print(" *  Writing to default location: ")

                path                = None;

                directory, filename = split(self.path);
                filename, extension = splitext(filename);

                if isdir(join(directory, 'QCM')):
                    path            = join(directory, 'QCM', str(filename + '_QCM' + extension));
                else:
                    mkdir(join(directory, 'QCM'));

                    if isdir(join(directory, 'QCM')):
                        path        = join(directory, 'QCM', str(filename + '_QCM' + extension));
                    else:
                        path        = join(directory, str(filename + '_QCM' + extension));

                print("    " + path + "\n");

            else:
                if splitext(self.QCM_path)[1] is '':
                    self.__QCM_path = self.QCM_path + ".vtk";

                path                = self.QCM_path;


            writer.SetFileName(path);

            for i in xrange(self.npoints):
                newPointData.InsertPoint(i, (self.QCM_points[0, i], self.QCM_points[1, i], 0.0));

            newPolyData.SetPoints(newPointData);
            newPolyData.SetPolys(self.polydata.GetPolys());
            if self.polydata.GetPointData().GetScalars() is None:
                newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetArray(0));
            else:
                newPolyData.GetPointData().SetScalars(self.polydata.GetPointData().GetScalars());

            writer.SetInputData(newPolyData);
            writer.Write();

            self.__QCM_polydata  = newPolyData;

            system("perl -pi -e 's/,/./g' %s " % path);

        else:
            raise RuntimeError("Information provided insufficient");

    def __calc_laplacian(self):
        numDims                 = self.polygons.shape[0];
        numPoints               = self.points.shape[1];
        numPolygons             = self.polygons.shape[1];

        sparseMatrix            = csr_matrix((numPoints, numPoints));

        for i in range(0, numDims):
            i1                  = (i + 0)%3;
            i2                  = (i + 1)%3;
            i3                  = (i + 2)%3;

            distP2P1            = self.points[:, self.polygons[i2, :]] - self.points[:, self.polygons[i1, :]];
            distP3P1            = self.points[:, self.polygons[i3, :]] - self.points[:, self.polygons[i1, :]];

            distP2P1            = distP2P1 / repmat(sqrt((distP2P1**2).sum(0)), 3, 1);
            distP3P1            = distP3P1 / repmat(sqrt((distP3P1**2).sum(0)), 3, 1);

            angles              = arccos((distP2P1 * distP3P1).sum(0));

            iterData1           = csr_matrix((1/tan(angles), 
                                                    (self.polygons[i2,:], 
                                                     self.polygons[i3,:])), 
                                                    shape=(numPoints, numPoints));

            iterData2           = csr_matrix((1/tan(angles), (self.polygons[i3,:], self.polygons[i2,:])), shape=(numPoints, numPoints));

            sparseMatrix        = sparseMatrix + iterData1 + iterData2;

        diagonal                = sparseMatrix.sum(0);
        diagonalSparse          = spdiags(diagonal, 0, numPoints, numPoints);
        self.__laplacian        = diagonalSparse - sparseMatrix;

    def __calc_QCM_points(self):
        if self.laplacian is not None:
            if self.boundary is not None:
                (nzi, nzj)      = find(self.laplacian)[0:2];

                for point in self.boundary:
                    positions   = where(nzi==point)[0];

                    self.laplacian[nzi[positions], nzj[positions]] = 0;

                    self.laplacian[point, point] = 1;

                angles                  = self.__calc_boundary_node_angles();
                Z                       = zeros((2, angles.size));
                Z[0,:]                  = cos(angles);
                Z[1,:]                  = sin(angles);

                boundaryConstrain = zeros((2, self.npoints));
                boundaryConstrain[:, self.boundary] = Z;

                self.__QCM_points = spsolve(self.laplacian, boundaryConstrain.transpose()).transpose();

                self.__calc_thin_plate_splines();

    def __calc_thin_plate_splines(self):
        if self.QCM_points is not None:
            if self.apex is not None:
                boundaryPoints  = self.QCM_points[:,self.boundary];
                source          = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));
                destination     = zeros((boundaryPoints.shape[0],
                                               boundaryPoints.shape[1] + 1));

                source[:, 0:source.shape[1] - 1]        = boundaryPoints;
                source[:, source.shape[1] - 1]          = self.QCM_points[:, self.apex];

                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;
                destination[:, 0:source.shape[1] - 1]   = boundaryPoints;

                x = source[0,:];
                y = source[1,:];
                d = destination[0,:] + 1j*destination[1,:];

                thinPlateInterpolation = RBFThinPlateSpline(x,y,d);

                result = thinPlateInterpolation(self.QCM_points[0,:], self.QCM_points[1,:]);

                self.__QCM_points[0,:] = result.real;
                self.__QCM_points[1,:] = result.imag;

    def __calc_boundary(self):
        startingPoint           = None;
        currentPoint            = None;
        foundBoundary           = False;
        cellId                  = None;
        boundary                = [];
        visitedEdges            = [];
        visitedBoundaryEdges    = [];

        for cellId in xrange(self.polydata.GetNumberOfCells()):
            cellPointIdList     = vtkIdList();
            cellEdges           = [];

            self.polydata.GetCellPoints(cellId, cellPointIdList);

            cellEdges           = [[cellPointIdList.GetId(0), 
                                    cellPointIdList.GetId(1)], 
                                   [cellPointIdList.GetId(1), 
                                    cellPointIdList.GetId(2)], 
                                   [cellPointIdList.GetId(2), 
                                    cellPointIdList.GetId(0)]];

            for i in xrange(len(cellEdges)):
                if (cellEdges[i] in visitedEdges) == False:
                    visitedEdges.append(cellEdges[i]);

                    edgeIdList  = vtkIdList();
                    edgeIdList.InsertNextId(cellEdges[i][0]);
                    edgeIdList.InsertNextId(cellEdges[i][1]);

                    singleCellEdgeNeighborIds = vtkIdList();

                    self.polydata.GetCellEdgeNeighbors(cellId, cellEdges[i][0], cellEdges[i][1], singleCellEdgeNeighborIds);

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

            self.polydata.GetPointCells(currentPoint, neighboringCells);

            for i in xrange(neighboringCells.GetNumberOfIds()):
                cell = neighboringCells.GetId(i);
                triangle = self.polydata.GetCell(cell);

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

                self.polydata.GetCellEdgeNeighbors(cell, edge1[0], edge1[1], edgeNeighbors1);

                self.polydata.GetCellEdgeNeighbors(cell, edge2[0], edge2[1], edgeNeighbors2);

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

        center      = mean(self.points[:,boundary], axis=1);
        vector1     = asarray(self.points[:,boundary[0]] - center);
        vector2     = asarray(self.points[:,boundary[1]] - center);
        vectorNormal= cross(vector1, vector2);
        vectorApex  = self.points[:, self.apex] - center;

        if len(center.shape) is not 1:
            if center.shape[0] is not 3:
                raise Exception("Something went wrong. Probably forgot to transpose this. Contact maintainer.");

        if dot(vectorApex, vectorNormal) < 0:
            boundary            = flipud(boundary);
            boundary            = roll(boundary, 1);

        self.__boundary         = boundary;

    def flip_boundary(self):
        self.__boundary         = flipud(self.boundary);
        self.__boundary         = roll(self.boundary, 1);

    def get_boundary_node_distances(self):
        boundaryNext            = roll(self.boundary, -1);
        boundaryNextPoints      = self.points[:, boundaryNext];

        distanceToNext          = boundaryNextPoints - self.boundary_points;

        return sqrt((distanceToNext**2).sum(0));

    def get_boundary_perimeter(self):
        return self.get_boundary_node_distances().sum();

    def get_boundary_node_distances_fraction(self):
        euclideanNorm           = self.get_boundary_node_distances();
        perimeter               = euclideanNorm.sum();

        return euclideanNorm/perimeter;

    def __calc_boundary_node_angles(self):
        circleLength            = 2*pi;
        fraction                = self.get_boundary_node_distances_fraction();

        angles                  = cumsum(circleLength*fraction);
        angles                  = roll(angles, 1);
        angles[0]               = 0;

        return angles;

    def __rearrange(self, objectivePoint=None):
        septalIndex             = None;
        septalPoint             = None;
        closestPoint            = None;

        if objectivePoint is None:
            if self.septum is None:
                raise Exception("No septal point provided in function call and no septal point provided in constructor. Aborting arrangement. ");
            else:
                septalIndex     = self.septum;
        else:
            print(" *  Using provided septal point as rearranging point.");
            self.__septum       = objectivePoint;
            septalIndex         = objectivePoint;

        if septalIndex in self.boundary:
            closestPoint        = septalIndex;
            closestPointIndex   = where(self.boundary==septalIndex);

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary     = roll(self.boundary, -closestPointIndex);
        else:
            try:
                septalPoint     = self.points[:, septalIndex];
            except:
                raise Exception("Septal point provided out of data bounds; the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.");

            if len(self.boundary.shape) == 1:
                septalPoint     = repmat(septalPoint,
                                    self.boundary.size, 1);
                septalPoint     = septalPoint .transpose();
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.");

            distanceToObjectivePoint    = (self.points[:, self.boundary] - septalPoint);
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0));
            closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min());

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            self.__boundary     = roll(self.boundary, -closestPointIndex);



    def rearrange_boundary(self, objectivePoint):
        """Rearranges the boundary aroung a new point identifier
        """
        old_septum              = self.septum;
        septalIndex             = None;
        closestPoint            = None;

        if objectivePoint == self.septum:
            return

        if objectivePoint in self.boundary:
            septalIndex         = objectivePoint;
            closestPoint        = objectivePoint;
            closestPointIndex   = where(self.boundary==objectivePoint);

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            # self.__boundary = roll(self.boundary, -septalIndex);
        else:
            print(" *  Provided point not found in the boundary. Selecting \n"
                  "    closest point available...");

            try:
                searched_point  = self.points[:, objectivePoint];
            except:
                raise Exception("Septal point provided out of data bounds; the point does not exist (it is out of bounds) or a point identifier beyond the total amount of points has been provided. Check input.");

            if len(self.boundary.shape) == 1:
                searched_point  = repmat(searched_point, self.boundary.size, 1);
                searched_point  = searched_point.transpose();
            else:
                raise Exception("It seems you have multiple boundaries. Contact the package maintainer.");

            distanceToObjectivePoint    = (self.points[:, self.boundary] - searched_point);
            distanceToObjectivePoint    = sqrt((distanceToObjectivePoint**2).sum(0));
            closestPointIndex           = where(distanceToObjectivePoint == distanceToObjectivePoint.min());

            if len(closestPointIndex) == 1:
                if len(closestPointIndex[0]) == 1:
                    closestPointIndex   = closestPointIndex[0][0];
                else:
                    raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");
            else:
                raise Exception("It seems your vtk file has more than one point ID associated to the objective point. Check your input data or contact the maintainer.");

            septalIndex                 = self.boundary[closestPointIndex];

        self.__boundary                 = roll(self.boundary, -closestPointIndex);

        center  = asarray([0, 0]); # The center of the disk will always be a 
        vector1 = asarray([1, 0]); # point (0,0) and the septum a vector (1,0), 
                                   # as induced by the boundary conditions
        vector2 = self.QCM_points[:, septalIndex] - center;

        angle   = arccos(dot(vector1, vector2)/(norm(vector1)*norm(vector2)));

        # If the y coordinate of the vector w.r.t. the rotation will take place
        # is negative, the rotation must be done counterclock-wise
        if vector2[1] > 0:
            angle = -angle;

        rotation_matrix                 = asarray([[cos(angle), -sin(angle)],
                                                   [sin(angle), cos(angle)]]);

        self.__septum                   = septalIndex;
        self.__QCM_points               = rotation_matrix.dot(self.QCM_points);

        self.__write_QCM_polydata();

    def __str__(self):
        s = "'" + self.__class__.__name__ + "' object at '" + self.path + "'.\n";
        s = s + "Number of dimensions: " + str(self.ndim) + "\n";
        s = s + "Number of points: " + str(self.npoints) + "\n";
        s = s + "Number of polygons: " + str(self.npolygons) + "\n";
        s = s + "Number of edges of the polygons: " + str(self.nedges_mesh) + "\n";
        s = s + "Scalar information: " + str(self.scalars_names);
        s = s + "Output file location: " + str(self.QCM_path);
        return s;










