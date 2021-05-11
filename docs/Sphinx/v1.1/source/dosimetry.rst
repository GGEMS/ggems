*********
Dosimetry
*********

During GGEMS simulation, a photon dosimetry module can be activated to compute absorbed dose in a specific phantom. 

.. NOTE:: Only photon are simulated in the current version of GGEMS. In next releases electron will be implemented.

The dosimetry module is 'GGEMSDosimetryCalculator':

.. code-block:: python

  dosimetry = GGEMSDosimetryCalculator()

After creating the GGEMSDosimetryCalculator object, a navigator is attached:

.. code-block:: python

  dosimetry.attach_to_navigator('phantom')

The size of voxel in dosimetry image (dosel) can be set. If not set the dosel size is the same than voxel phantom size:

.. code-block:: python

  dosimetry.set_dosel_size(0.5, 0.5, 0.5, 'mm')

The absorbed dose is computed in gray (Gy). By default the dose in computed using materials in phantom. Otherwize the user can set water material everywhere in phantom.

.. code-block:: python

  dosimetry.water_reference(True)

A custom threshold can be set on density. If density of phantom is below the threshold the dose value in 0.

.. code-block:: python

  dosimetry.minimum_density(0.1, 'g/cm3')

For dose output, there are many informations the user can save such as: uncertainty value of the dose, the deposited energy in dosel, the squared of deposited energy in dosel and the number of interaction (hit) in dosel:

.. code-block:: python

  dosimetry.set_output('data/dosimetry')
  dosimetry.uncertainty(True)
  dosimetry.edep(True)
  dosimetry.hit(True)
  dosimetry.edep_squared(True)

There is a special output named 'photon tracking'. This output registers the number of photons crossing a dosel. To use this option, the size of dosel has to be the same than the phantom voxel size, otherwize GGEMS will throw an error:

.. code-block:: python

  dosimetry.photon_tracking(True)
