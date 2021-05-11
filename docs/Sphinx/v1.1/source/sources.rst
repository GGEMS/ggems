*******
Sources
*******

The source strategy in GGEMS is to develop a optimized source for each application.
For moment, only CT/CBCT application is developed so the source type available is
a cone-beam X-ray source.

X-ray Source
============
X-ray source is defined as a cone-beam geometry. The direction of the generated
particles point always to the center of the world. This source has its own axis
as defined in the image below:

.. figure:: ../images/source.png
    :width: 50%
    :align: center

Some commands are provided managing a X-ray source.

First, the user has to create source by choosing a name:

.. code-block:: python

  xray_source = GGEMSXRaySource('xray_source')

The particle type is only photon and can be selected with the following command:

.. code-block:: python

  xray_source.set_source_particle_type('gamma')

The number of generated particles during the run is defined by the user:

.. code-block:: python

  xray_source.set_number_of_particles(1000000000)

The position and rotation of the source are defined in the global world reference axis
and the cone-beam source is defined with an aperture angle.

.. code-block:: python

  xray_source.set_position(-595.0, 0.0, 0.0, 'mm')
  xray_source.set_rotation(0.0, 0.0, 0.0, 'deg')
  xray_source.set_beam_aperture(12.5, 'deg')

A X-ray source is defined with a focal spot size. If defined at (0, 0, 0) mm, it
is similar to a point source, otherwize it is a more realistic X-ray source with
a small rectangular surface defined in source axis reference:

.. code-block:: python

  xray_source.set_focal_spot_size(0.0, 0.0, 0.0, 'mm')

.. IMPORTANT::

  The focal spot size is defined in source axis reference and not in global world
  reference!!!

The energy source can be defined using a single energy value or a spectrum included in
a text file.

.. code-block:: python

  xray_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')
  # OR
  xray_source.set_monoenergy(25.0, 'keV')
