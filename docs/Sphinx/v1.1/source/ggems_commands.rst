**************
GGEMS Commands
**************

The main steps in GGEMS are initialize and run methods.

.. code-block:: python

  ggems = GGEMS()
  ggems.initialize()
  ggems.run()

Different useful information output are available for the user during GGEMS executions.

To print all informations about OpenCL device:

.. code-block:: python

  ggems.opencl_verbose(True)

To print all informations about material database:

.. code-block:: python

  ggems.material_database_verbose(True)

To print all information about navigator (system + phantom):

.. code-block:: python

  ggems.navigator_verbose(True)

To print all informations about source:

.. code-block:: python

  ggems.source_verbose(True)

To print all informations about allocated memory:

.. code-block:: python

  ggems.memory_verbose(True)

To print all informations about activated processes:

.. code-block:: python

  ggems.process_verbose(True)

To print all informations about range cuts:

.. code-block:: python

  ggems.range_cuts_verbose(True)

To print seed and state of the random:

.. code-block:: python

  ggems.random_verbose(True)

To print profiler data (elapsed time in OpenCL kernels):

.. code-block:: python

  ggems.profiling_verbose(True)

To print tracking informations about a specific particle index:

.. code-block:: python

  ggems.tracking_verbose(True, 12)

Cleaning GGEMS object

.. code-block:: python

  ggems.delete()