Basics
^^^^^^

In order to build any synthetic vascular network(s), you will need to define at least
three components:

* the perfusion domain in which the network(s) is built
* seed point(s) where the vascular network(s) is initiated
* the size of the vascular network(s)

By default if no seed points are given, vascular networks will be randomly seeded
along the surface boundary of the perfusion domain. Likewise if no tree size is given,
vascular networks will be initiated and dynamically grown. Therefore if the seed
points and/or size of the vascular networks are unknown, the toolkit will make
basic assumptions on the backend to fill in the missing information. The only
component SVT cannot assume is the perfusion domain. This *must* be supplied by
the user.

Defining a Perfusion Domain
===========================

In this section, we will cover the basics of creating a perfusion domain object
necessary for generating a vascular network with SVT. The most basic method for
defining a perfusion domain relies on point-wise data along the boundary. To create
an example domain, we will use `Pyvista <https://docs.pyvista.org/>`_ to obtain
boundary points and normals for a simple cube volume.

.. code-block:: python
    :linenos:

    import pyvista as pv
    import svcco

    # Define some set of points along a cube boundary
    cube = pv.Cube().triangulate().subdivide(4)

    # Create a domain instance
    surf = svcco.surface()

    # Assign points and normals to the domain object
    surf.set_data(cube.points,cube.point_normals)

    # Solve the domain geometry to obtain an implicit representation
    surf.solve()

    # Build the implicit domain
    surf.build(q=3)

To visualize the perfusion domain object, we can call ``svcco.plot_volume(surf)``
to render some number of isocontours for the implicit funciton defining our domain.
Below shows only one isocontour of the implicit cube domain near the boundary of the
domain.

.. raw:: html
   :file: ../build/html/_plots/cube_with_normals.html

In this example, we provided the point-wise normal vectors associated with the
cube since we already know the normals for our domain boundary. It is *not always*
the case that normal information will be available. Therefore, point-wise normal
vectors *do not* need to be provided by the user in order to create perfusion domain.
However, providing them allows for fast solving of the implicit function defining
the domain and can generally improve accuracy at the boundary if a domain has sharp edges.

.. note::
    **Providing normal vector information at points is not required** for perfusion
    domain objects, but it can improve performance during the ``solve()`` and
    ``build()`` stages, especially if the domain is only C0 continuous.

We can demonstrate this this same process without including normal vector information
below:

.. code-block:: python
    :linenos:

    import pyvista as pv
    import svcco

    # Define some set of points along a cube boundary
    cube = pv.Cube().triangulate().subdivide(4)

    # Create a domain instance
    surf = svcco.surface()

    # Assign points and normals to the domain object
    surf.set_data(cube.points)

    # Solve the domain geometry to obtain an implicit representation
    surf.solve()

    # Build the implicit domain
    surf.build(q=3)

.. raw:: html
   :file: ../build/html/_plots/cube_without_normals.html

Of course, defining perfusion domains through point arrays can be cumbersome and
not immediately available for many geometry files, especially STL files which are
common for 3D fabrication. Often complex geometries are represented as mesh files
of varying quality. To handle these types of files for perfusion domain creation,
we can use the ``load()`` method to import and clean a mesh geometry file prior to
implicit domain creation.

.. code-block::
   :linenos:

   import svcco

   # Define a domain instance
   surf = svcco.surface()

   # Use the load method to import a geometry
   surf.load('/path/to/file')

   # Solve the domain geometry to obtain an implicit representation
   surf.solve()

   # Build the implicit domain
   surf.build(q=3)

.. note::
   If the ``load()`` method accepts a string parameter for a file path to supported
   mesh files. If no arguments are provided, the method will open a file navigator window
   for the user to select a file through an easy-to-use interface.

For more information and advanced features of the perfusion domain ``svcco.surface()``
class, please refer to the documentation in :doc:`implicit <svcco.implicit>` module.
