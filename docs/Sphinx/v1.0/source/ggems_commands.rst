**************
GGEMS Commands
**************

The main steps in GGEMS are the initialize and run methods.

.. code-block:: python

  ggems_manager.initialize()
  ggems_manager.run()

Different useful information output are available for the user during GGEMS executions.

To print all informations about OpenCL device:

.. code-block:: python

  ggems_manager.opencl_verbose(True)

To print all informations about material database:

.. code-block:: python

  ggems_manager.material_database_verbose(True)

To print all information about navigator (system + phantom):

.. code-block:: python

  ggems_manager.navigator_verbose(True)

To print all informations about source:

.. code-block:: python

  ggems_manager.source_verbose(True)

To print all informations about allocated memory:

.. code-block:: python

  ggems_manager.memory_verbose(True)

To print all informations about activated processes:

.. code-block:: python

  ggems_manager.process_verbose(True)

To print all informations about range cuts:

.. code-block:: python

  ggems_manager.range_cuts_verbose(True)

To print seed and state of the random:

.. code-block:: python

  ggems_manager.random_verbose(True)

To print kernel computation time:

.. code-block:: python

  ggems_manager.kernel_verbose(True)

To print tracking informations about a specific particle index:

.. code-block:: python

  ggems_manager.tracking_verbose(True, 12)
