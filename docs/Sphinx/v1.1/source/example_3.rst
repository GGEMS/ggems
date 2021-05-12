***************************************
Examples 3: Voxelized Phantom Generator
***************************************

A tool creating voxelized phantom is provided by GGEMS. Only basic shapes are available such as tube, box and sphere. The output format is MHD, and the range material data file is created in same time than the voxelized volume.

.. code-block:: console

  $ python generate_volume.py [-h] [-d DEVICE] [-v VERBOSE]
  -h/--help           Printing help into the screen
  -d/--device         Setting OpenCL id
  -v/--verbose        Setting level of verbosity

First step is to create global volume storing all other voxelized objets. Dimension, voxel size, name of output volume, format data type and material are defined.

.. code-block:: python

  volume_creator_manager.set_dimensions(450, 450, 450)
  volume_creator_manager.set_element_sizes(0.5, 0.5, 0.5, "mm")
  volume_creator_manager.set_output('data/volume')
  volume_creator_manager.set_range_output('data/range_volume')
  volume_creator_manager.set_material('Air')
  volume_creator_manager.set_data_type('MET_INT')
  volume_creator_manager.initialize()

Then a voxelized volume can be drawn in the global volume. A box object is built with the command lines below:

.. code-block:: python

  box = GGEMSBox(24.0, 36.0, 56.0, 'mm')
  box.set_position(-70.0, -30.0, 10.0, 'mm')
  box.set_label_value(1)
  box.set_material('Water')
  box.initialize()
  box.draw()
  box.delete()
