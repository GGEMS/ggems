*****************************
Examples 1: Total Attenuation
*****************************

.. WARNING::

  This example is only available using python and the matplotlib library is mandatory.

This example is a tool for plotting the total attenuation of a material for energy between 0.01 MeV and 1 MeV. The commands are similar to example 0, and all physical processes are activated.

.. code-block:: console

  $ python total_attenuation.py [-h] [-d DEVICE] [-m MATERIAL] [-v VERBOSE]
  -h/--help           Printing help into the screen
  -d/--device         Setting OpenCL id
  -m/--material       Setting one of material defined in GGEMS (Water, Air, ...)
  -v/--verbose        Setting level of verbosity

Total attenuations for Water and LSO are shown below:

.. image:: ../images/Water_Total_Attenuation.png
  :width: 800
  :align: center

.. image:: ../images/LSO_Total_Attenuation.png
  :width: 800
  :align: center