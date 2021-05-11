*****
World
*****

Outside navigator, particle are not tracked. However, a tool has been developped in GGEMSWorld class storing particle data (photon tracking, energy/energy squared voxel in world, and momentum) outside navigator. Particles are projected in GGEMSWorld using a DDA algorithm.

The world module is 'GGEMSWorld':

.. code-block:: python

  world = GGEMSWorld()

After creating the GGEMSWorld object, the dimension of the world and size of voxel can be set:

.. code-block:: python

  world.set_dimensions(200, 200, 200)
  world.set_element_sizes(10.0, 10.0, 10.0, 'mm')

For world output, there are many informations the user can save such as: energy and energy squared of photon crossing voxel, photon momentum and fluence (photon tracking):

.. code-block:: python

  world.set_output_basename('data/world')
  world.energy_tracking(True)
  world.energy_squared_tracking(True)
  world.momentum(True)
  world.photon_tracking(True)
