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


# from __future__ import division

# import os;
# import numpy;
# import scipy;
# import mvpoly.rbf;
# import vtk;

# from numpy.matlib import repmat;
# from numpy import int;
# from scipy import zeros;
# from scipy import asarray;
# from scipy import mean;
# from scipy import sqrt;
# from scipy import pi;
# from scipy import sin;
# from scipy import cos;
# from scipy import tan;
# from scipy import arccos;
# from scipy import cumsum;
# from scipy import where;
# from scipy import roll;
# from scipy import flip;
# # from scipy.interpolate import Rbf;
# # from mvpoly.rbf import RBFThinPlateSpline;

# from scipy.sparse import csr_matrix;
# from scipy.sparse import spdiags;
# from scipy.sparse.linalg import spsolve; 




# # import scipy, time, os, numpy, VentricularImage;
# from abc import ABCMeta, abstractmethod;

# class wardrobe(object):
#     __metaclass__ = ABCMeta;

#     namme = None;
#     amount1 = None;
#     amount2 = None;

#     @abstractmethod
#     def __init__(self, number1, number2=56):
#         pass;

#     @classmethod
#     def GetName(self):
#         return self.namme;

#     @abstractmethod
#     def SetName(self, name):
#         pass;

#     @property
#     def amount2(self):
#         return self.amout1;

#     @amount1.setter
#     def amount2(self, value):
#         print("AAAAAAAAAAAA");
#         self.amount1 = value;


# class waWardrobe(wardrobe):
#     __clothes = None;

#     def __init__(self, name, number1, cloth, number2=56):
#         s              = super(wardrobe, self).__init__();
#         self.namme = name;
#         self.amount1 = number1;
#         self.amount2 = number2;
#         self.__clothes = cloth;

#     @property
#     def amount1(self):
#         print("Getting...");
#         return self.amount1;

#     @amount1.setter
#     def amount1(self, value):
#         print("Setting")
#         self.amoun1 = value;

#     def SetName(self):
#         self.namme = name;


# B = waWardrobe("Michaels", 2,5,6);
# B.GetName();




    # @classmethod;
    # def 

def main():
    # septum_MRI = 201479 - 1;
    # apex_MRI = 37963 - 1;
    # path_MRI = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

    # apex_EAM = 599 - 1;
    # septum_EAM = 1389 - 1;
    # path_EAM = os.path.join("/home/guille/BitBucket/qcm/data/pat1/EAM", "pat1_EAM_endo_smooth.vtk");

    # start = time.time(); reload(VentricularImage); MRI = VentricularImage.VentricularImage(path_MRI, septum_MRI, apex_MRI); print(time.time() - start);
    # start = time.time(); reload(VentricularImage); EAM = VentricularImage.VentricularImage(path_EAM, septum_EAM, apex_EAM); print(time.time() - start);

    # start = time.time(); reload(VentricularInterpolator); interpolator = VentricularInterpolator.VentricularInterpolator(MRI, EAM); print(time.time() - start);



import scipy, time, os, numpy, VentricularImage;

septum_MRI = 201479 - 1;
apex_MRI = 37963 - 1;
path_MRI = os.path.join("/home/guille/BitBucket/qcm/data/pat1/MRI", "pat1_MRI_Layer_6.vtk");

start = time.time(); reload(VentricularImage); MRI = VentricularImage.VentricularImage(path_MRI, septum_MRI, apex_MRI); print(time.time() - start);


apex_EAM = 599 - 1;
septum_EAM = 1389 - 1;
path_EAM = os.path.join("/home/guille/BitBucket/qcm/data/pat1/EAM", "pat1_EAM_endo_smooth.vtk");

start = time.time(); reload(VentricularImage); EAM = VentricularImage.VentricularImage(path_EAM, septum_EAM, apex_EAM); print(time.time() - start);

start = time.time(); reload(VentricularInterpolator); interpolator = VentricularInterpolator.VentricularInterpolator(MRI, EAM); interpolator.kNearestNeighbours(); print(time.time() - start);






if __name__ == '__main__':
    main();









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



