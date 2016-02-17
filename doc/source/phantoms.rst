.. GGEMS documentation: Phantoms

Phantoms
========

Two main phantoms are used in GGEMS. Each one handle different particle navigation dedicated to a specific application. For instance a phantom *VoxPhanImgNav* is used for CT imaging while a phantom *VoxPhanDosiNav* is used in radiotherapy application. 

+----------------+---------------------+---------------------+---------------------+-----------------+
| Phantom name   | Type of volume      | Particle navigator  | Specificity         | Applications    |
+================+=====================+=====================+=====================+=================+
| VoxPhanImgNav  | Voxelized           | Photon              | Record scattering   | Imaging         |
+----------------+---------------------+---------------------+---------------------+-----------------+
| VoxPhanDosiNav | Voxelized           | Photon and electron | Record dose         | Dosimetry       |
+----------------+---------------------+---------------------+---------------------+-----------------+


VoxPhanImgNav
-------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert

This class handle particle navigation within a voxelized volume. This phantom consider only photon particle and the dose within the volume is not recorded. This phantom and the associated particle navigator is mainly used for imaging application such CT.

------------

.. c:function:: void load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
    
    Load a voxelized phantom from a MHD format image (``TODO: MHD spec``). This file can content Hounsfield units (HU) or simple material ID. To convert material ID or HU into material a range material file is required. This file contains for each range of value (ID or HU) the material to be associated. Material name must be defined into the material database (``TODO: see GGEMS``).

    .. c:var:: filename  
        
        Filename of the mhd data file.

    .. c:var:: range_mat_name 
    
        Filename of the range material data file. 

Example of range material contains::

    0   100 Water
    101 200 Air
    201 220 RibBone    

-----

.. c:function:: void set_materials( std::string filename )

    Load the material database used by the range material data file and the phantom.

    .. c:var:: filename

        Filename of the material database. 

Example of material database contains ::

    # Water    
    Water: d=1.00 g/cm3; n=2;
        +el: name=Hydrogen  ; f=0.111
        +el: name=Oxygen    ; f=0.889

-----

.. note::
    Version: beta - Work for authors.

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    VoxPhanImgNav *aPhantom = new VoxPhanImgNav;
    aPhantom->load_phantom_from_mhd( "data/patient.mhd", "data/HU2mat.dat" );
    aPhantom->load_materials( "data/materials.dat" );




Last update: |today|  -  Release: |release|.