# -*- coding: utf-8 -*-

"""
    Copyright (C) 2017 - Universitat Pompeu Fabra
    Authors - Guillermo Jimenez-Perez & Constantine Butakoff 
    Correspondence - Guillermo Jimenez⁻Perez <guillermo.jim.per@gmail.com>

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

        try:
            for i in range(0, len(self.pointIds)):
                self.selected_point_ids.InsertNextId(self.pointIds[i])
                self.selected_points.InsertNextPoint(shape.GetPoint(self.pointIds[i]))
        except:
            raise Exception("pointIds has to be iterable")


    def GetSelectedPointIds(self): #returns vtkIdList in the order of clicks
        return self.selected_point_ids
        
    def GetSelectedPoints(self): #returns vtkPoints in the order of clicks
        return self.selected_points
        
    def DoSelection(self, shape): #open rendering window and start 
        if self.pointIds is None:
            self.selected_points.Reset()
            self.selected_point_ids.Reset()
        else:
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

            # #create a sphere to mark the location
            # sphereSource = vtkSphereSource();
            # sphereSource.SetRadius(self.marker_radius); 
            # sphereSource.SetCenter(self.selected_points.GetPoint(0));
            
            # mapper = vtkPolyDataMapper()
            # mapper.SetInputConnection(sphereSource.GetOutputPort())

            # actor = vtkActor()
            # actor.SetMapper(mapper)

            # #define the color of the sphere (pick from the list)
            # actor.GetProperty().SetColor(self.marker_colors[0])

            # renderer = vtkRenderer();
            # renderer.AddActor(actor)

            # #create a sphere to mark the location
            # sphereSource = vtkSphereSource();
            # sphereSource.SetRadius(self.marker_radius); 
            # sphereSource.SetCenter(self.selected_points.GetPoint(1));
            
            # mapper = vtkPolyDataMapper()
            # mapper.SetInputConnection(sphereSource.GetOutputPort())

            # actor = vtkActor()
            # actor.SetMapper(mapper)

            # #define the color of the sphere (pick from the list)
            # actor.GetProperty().SetColor(self.marker_colors[1])

            # renderer.AddActor(actor)


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

