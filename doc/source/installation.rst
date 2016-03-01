.. GGEMS documentation: Compilation

.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert


Compilation
===========

Requirement
-----------

Software using GGEMS library requires minimum version of gcc and nvcc to be compiled:

GCC
^^^

GGEMS library requires a **gcc** version **>= 4.8.4**.

NVCC
^^^^

GGEMS library requires a **nvcc** version **>= 7.0.27**.

Compute capability
^^^^^^^^^^^^^^^^^^

Your NVIDIA GPU device must have a compute capability **>= 3.0**.

Only Kepler and newest NVIDIA architecture are compatible with GGEMS. Check your architecture to `Wikipedia <http://en.wikipedia.org.wiki/CUDA>`_.

Installation
------------

Since GGEMS is a static library, installation procedure is not really required. The main directory that contains GGEMS has to be placed somewhere in your home directory in order to have the right to access.

Compilation
-----------

Your source code must include the main GGEMS file:

.. code-block:: cpp
    :linenos:

    #include <ggems.cuh>

Then in a linux terminal, you need to source GGEMS path in order to tell your compilation where is GGEMS files::

    source [MyGGEMSInstallDirectoryPath]/bin/ggems.sh

Finally, you can compile::

    make

Execution
^^^^^^^^^

Software that uses GGEMS library requires a license to be executed. License file can be freely distributed for academic institution.


Last update: |today|  -  Release: |release|.