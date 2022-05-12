import vtk
from vtk.util import numpy_support

def extract_surface(msh):
    msh_vtk = numpy_support.numpy_to_vtk(msh)
    pts = vtk.vtkPoints()
    pts.SetData(msh_vtk)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    bounds = poly.GetBounds()
    x_range = bounds[1]-bounds[0]
    y_range = bounds[3]-bounds[2]
    z_range = bounds[5]-bounds[4]

    PCA = vtk.vtkPCANormalEstimation()
    PCA.SetInputData(poly)
    PCA.SetSampleSize(10)
    PCA.SetNormalOrientationToGraphTraversal()
    PCA.FlipNormalsOn()

    distance = vtk.vtkSignedDistance()
    distance.SetInputConnection(PCA.GetOutputPort())
    distance.SetRadius(1)
    #distance.SetDimensions(256,256,256)
    distance.SetBounds(bounds[0] - x_range * .1, bounds[1] + x_range * .1,
                       bounds[2] - y_range * .1, bounds[3] + y_range * .1,
                       bounds[4] - z_range * .1, bounds[5] + z_range * .1)
    surf_extract = vtk.vtkExtractSurface()
    surf_extract.SetInputConnection(distance.GetOutputPort())
    surf_extract.SetRadius(1)
    surf_extract.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(surf_extract.GetOutputPort())
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def geodesic(polydata,proximal,distal,terminal,bifurcation):

    proximal_idx     = polydata.FindPoint(proximal)
    distal_idx       = polydata.FindPoint(distal)
    terminal_idx     = polydata.FindPoint(terminal)
    bifurcation_idx  = polydata.FindPoint(bifurcation)

    DIJ_1 = vtk.vtkDijkstraGraphGeodesicPath()
    DIJ_1.SetInputData(polydata)
    DIJ_1.SetStartVertex(proximal_idx)
    DIJ_1.SetEndVertex(bifurcation_idx)
    DIJ_1.Update()

    DIJ_2 = vtk.vtkDijkstraGraphGeodesicPath()
    DIJ_2.SetInputData(polydata)
    DIJ_2.SetStartVertex(bifurcation_idx)
    DIJ_2.SetEndVertex(distal_idx)
    DIJ_2.Update()

    DIJ_3 = vtk.vtkDijkstraGraphGeodesicPath()
    DIJ_3.SetInputData(polydata)
    DIJ_3.SetStartVertex(bifurcation_idx)
    DIJ_3.SetEndVertex(terminal_idx)
    DIJ_3.Update()


    poly_path_1 = DIJ_1.GetOutput()
    poly_path_2 = DIJ_2.GetOutput()
    poly_path_3 = DIJ_3.GetOutput()

    proximal_geodesic  = numpy_support.vtk_to_numpy(poly_path_1.GetPoints().GetData())
    distal_geodesic    = numpy_support.vtk_to_numpy(poly_path_2.GetPoints().GetData())
    terminal_geodesic  = numpy_support.vtk_to_numpy(poly_path_3.GetPoints().GetData())

    proximal_length = poly_path_1.GetLength()
    distal_length   = poly_path_2.GetLength()
    terminal_length = poly_path_3.GetLength()

    proximal_data = {'path'  :proximal_geodesic,
                     'length':proximal_length}

    distal_data   = {'path'  :distal_geodesic,
                     'length':distal_length}

    terminal_data = {'path'  :terminal_geodesic,
                     'length':terminal_length}

    return proximal_data,distal_data,terminal_data
