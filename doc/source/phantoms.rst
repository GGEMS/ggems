.. GGEMS documentation: Phantoms

.. _phantoms-label:

.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert

Phantoms
========

Three main phantoms are used in GGEMS. Each of them handles different particle navigation dedicated to a specific application. For instance a phantom *VoxPhanImgNav* is used for CT imaging while a phantom *VoxPhanDosiNav* is used for radiotherapy applications. 

+----------------+---------------------+---------------------+---------------------+-----------------+
| Phantom name   | Type of volume      | Particle navigator  | Specificity         | Applications    |
+================+=====================+=====================+=====================+=================+
| VoxPhanImgNav  | Voxelized           | Photon              | Record scattering   | Imaging         |
+----------------+---------------------+---------------------+---------------------+-----------------+
| VoxPhanDosiNav | Voxelized           | Photon and electron | Record dose         | MeV Dosimetry   |
+----------------+---------------------+---------------------+---------------------+-----------------+
| VoxPhanIORTNav | Voxelized           | Photon              | Record dose, TLE    | keV Dosimetry   |
+----------------+---------------------+---------------------+---------------------+-----------------+


VoxPhanImgNav
-------------

This class handles particle navigation within a voxelized volume. This phantom considers only photon particles and the dose values within the volume are not recorded. Such a phantom and its associated particle navigator are mainly used for imaging applications such as CT.

------------

.. c:function:: void load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
    
    Load a voxelized phantom from a MHD format image ( :ref:`mhd-label` ). This file can contain Hounsfield units (HU) or simple materials IDs. To convert a material ID or HU into material, a range material file is required. Such a file contains for each range of values (ID or HU) the material to be associated. Material's names must be defined in the material database.

    .. c:var:: filename  
        
        Filename of the mhd data file.

    .. c:var:: range_mat_name 
    
        Filename of the range material data file. 

Example of the contents of a range material file::

    0   100 Water
    101 200 Air
    201 220 RibBone    

-----

.. c:function:: void set_materials( std::string filename )

    Load the material database used by the range material data file and the phantom.

    .. c:var:: filename

        Filename of the material database. 

Example of the contents of a material database file ::

    # Water    
    Water: d=1.00 g/cm3; n=2;
        +el: name=Hydrogen  ; f=0.111
        +el: name=Oxygen    ; f=0.889

-----

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    VoxPhanImgNav *aPhantom = new VoxPhanImgNav;
    aPhantom->load_phantom_from_mhd( "data/patient.mhd", "data/HU2mat.dat" );
    aPhantom->load_materials( "data/materials.dat" );



VoxPhanDosiNav
--------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert

This class handles particle navigation within a voxelized volume. This phantom considers photon and electron particles and the dose values within the volume are recorded. Such a phantom and its associated particle navigator are mainly used for MeV dosimetry applications.

------------

.. c:function:: void load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
    
    Load a voxelized phantom from a MHD format image ( :ref:`mhd-label` ). This file can contain Hounsfield units (HU) or simple materials IDs. To convert a material ID or HU into material, a range material file is required. Such a file contains for each range of values (ID or HU) the material to be associated. Material's names must be defined in the material database.

    .. c:var:: filename  
        
        Filename of the mhd data file.

    .. c:var:: range_mat_name 
    
        Filename of the range material data file. 

Example of the contents of a range material file::

    0   100 Water
    101 200 Air
    201 220 RibBone    

-----

.. c:function:: void set_materials( std::string filename )

    Load the material database used by the range material data file and the phantom.

    .. c:var:: filename

        Filename of the material database. 

Example of the contents of a material database file::

    # Water    
    Water: d=1.00 g/cm3; n=2;
        +el: name=Hydrogen  ; f=0.111
        +el: name=Oxygen    ; f=0.889

-----

.. c:function:: void set_doxel_size( f32 sizex, f32 sizey, f32 sizez )

    Set the voxel size of the dose map. If no values are specified, the doxel size is the same
        as the phantom voxel size.

    .. c:var:: sizex

        Size (in mm) of the doxel along x-axis dimension. 

    .. c:var:: sizey

        Size (in mm) of the doxel along y-axis dimension. 

    .. c:var:: sizez

        Size (in mm) of the doxel along z-axis dimension.         

-----

.. c:function:: void set_volume_of_interest( f32 xmin, f32 xmax, f32 ymin, f32 ymax, f32 zmin, f32 zmax )

    Set a volume of interest (VOI) to record the dose within the phantom. This volume is defined according to the phantom offset i.e. the center of the world frame. If no values are specified, the volume of interest is set to consider the whole phantom volume. 

    .. c:var:: xmin

        Min position of VOI boundaries along x-axis of the volume of interest. 

    .. c:var:: xmax

        Max position of VOI boundaries along x-axis of the volume of interest. 

    .. c:var:: ymin

        Min position of VOI boundaries along y-axis of the volume of interest. 

    .. c:var:: ymax

        Max position of VOI boundaries along y-axis of the volume of interest. 

    .. c:var:: zmin

        Min position of VOI boundaries along z-axis of the volume of interest. 

    .. c:var:: zmax

        Max position of VOI boundaries along z-axis of the volume of interest.                                         

.. note::
    Version: alpha - Never test

-----

.. c:function:: void export_density_map( std::string filename )

    Export phantom density values to a MHD file.

    .. c:var:: filename

        Filename of the MHD file. 

-----

.. c:function:: void export_materials_map( std::string filename )

    Export phantom materials values after being labelled to a MHD file.

    .. c:var:: filename

        Filename of the MHD file. 

-----

.. c:function:: void calculate_dose_to_water()

    After the simulation, energies deposited within the volume are converted to dose using this function. The dose is caculated using water density.

-----

.. c:function:: void calculate_dose_to_phantom()

    After the simulation, energies deposited within the volume are converted to dose using this function. The dose is caculated using the density of each phantom voxel.

-----

.. c:function:: void write ( std::string filename = "dosimetry.mhd" )

    After dose calculation, the dosemap can be exported in MetaImage format ( :ref:`mhd-label` ). This function writes four files related to the dosemap:

    * xxx-Dose: final dose map in Gray
    * xxx-Edep: Deposited energy within the phantom in MeV
    * xxx-Hit: Number of hits within the phantom
    * xxx-Uncertainty: Dose uncertainty in %

    .. c:var:: filename

        Base filename used to export the data.        

-----

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    VoxPhanImgNav *aPhantom = new VoxPhanImgNav;
    aPhantom->load_phantom_from_mhd( "data/patient.mhd", "data/HU2mat.dat" );
    aPhantom->load_materials( "data/materials.dat" );


VoxPhanIORTNav
--------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert

This class handles particle navigation within a voxelized volume. This phantom considers only photon particles and the dose values within the volume are recorded using Track Length Estimator. Such a phantom and its associated particle navigator are mainly used for Intra-Operative Radiation Therapy (IORT) or keV external radiotherapy.

------------

.. c:function:: void load_phantom_from_mhd ( std::string filename, std::string range_mat_name )
    
    Load a voxelized phantom from a MHD format image ( :ref:`mhd-label` ). This file can contain Hounsfield units (HU) or simple materials IDs. To convert a material ID or HU into material, a range material file is required. Such a file contains for each range of values (ID or HU) the material to be associated. Material's names must be defined in the material database.

    .. c:var:: filename  
        
        Filename of the mhd data file.

    .. c:var:: range_mat_name 
    
        Filename of the range material data file. 

Example of the contents of a range material file::

    0   100 Water
    101 200 Air
    201 220 RibBone    

-----

.. c:function:: void set_materials( std::string filename )

    Load the material database used by the range material data file and the phantom.

    .. c:var:: filename

        Filename of the material database. 

Example of the contents of a material database file::

    # Water    
    Water: d=1.00 g/cm3; n=2;
        +el: name=Hydrogen  ; f=0.111
        +el: name=Oxygen    ; f=0.889

-----

.. c:function:: void set_kerma_estimator( std::string kind )

    This kerma estimator can be used only while energies of primary particles are below 1 MeV. It is possible to use the Track Length Estimator (TLE) for a fast dosimetry. 

    .. c:var:: kind

        Name of the kerma estimator, can be ``"TLE"`` or ``"Analog"``. Default value is ``"Analog"``. 

-----

.. c:function:: void export_density_map( std::string filename )

    Export phantom density values to a MHD file.

    .. c:var:: filename

        Filename of the MHD file. 

-----

.. c:function:: void export_materials_map( std::string filename )

    Export phantom materials values after being labelled to a MHD file.

    .. c:var:: filename

        Filename of the MHD file. 

-----

.. c:function:: void calculate_dose_to_water()

    After the simulation, energies deposited within the volume are converted to dose using this function. The dose is caculated using water density.

-----

.. c:function:: void calculate_dose_to_phantom()

    After the simulation, energies deposited within the volume are converted to dose using this function. The dose is caculated using the density of each phantom voxel.

-----

.. c:function:: void write ( std::string filename = "dosimetry.mhd" )

    After dose calculation, the dosemap can be exported in MetaImage format ( :ref:`mhd-label` ). This function writes four files related to the dosemap:

    * xxx-Dose: final dose map in Gray
    * xxx-Edep: Deposited energy within the phantom in MeV
    * xxx-Hit: Number of hits within the phantom
    * xxx-Uncertainty: Dose uncertainty in %

    .. c:var:: filename

        Base filename used to export the data.        

-----

.. note::
    Dosemap has the same sizes than the voxelized phantom and the same voxel sizes.

-----

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    VoxPhanIORTNav *aPhantom = new VoxPhanIORTNav;
    aPhantom->load_phantom_from_mhd( "data/patient.mhd", "data/HU2mat.dat" );
    aPhantom->load_materials( "data/materials.dat" );
    aPhantom->set_kerma_estimator( "TLE" );



Last update: |today|  -  Release: |release|.
