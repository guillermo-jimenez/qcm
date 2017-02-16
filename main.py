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



import os;
import vtk;
import ConfigParser;


Config = ConfigParser.ConfigParser();

Config.read("c:\\tomorrow.ini");




for subdir, dirs, files in os.walk(rootdir):
    for file in files:









def docstring_example():
    """ 
        Multi-line Docstrings

        Multi-line docstrings consist of a summary line just like a one-line
        docstring, followed by a blank line, followed by a more elaborate
        description. The summary line may be used by automatic indexing tools;
        it is important that it fits on one line and is separated from the rest
        of the docstring by a blank line. The summary line may be on the same
        line as the opening quotes or on the next line. The entire docstring is
        indented the same as the quotes at its first line (see example below).

        Example:
        def complex(real=0.0, imag=0.0):
            '''
                Form a complex number.

                Keyword arguments:
                real -- the real part (default 0.0)
                imag -- the imaginary part (default 0.0)
                ...
            '''


    """


def reader_vtk():
    reader = vtk.vtkXMLPolyDataReader();
    path = os.path.join(constant.BASE_DIR, "archive.vtp"); #Archive path
    reader.SetFileName(path);
    reader.Update();
    
    mapper = vtk.vtkPolyDataMapper();
    mapper.SetInput(reader.GetOutput());
    
    actor = vtk.vtkActor();
    actor.SetMapper(mapper);





