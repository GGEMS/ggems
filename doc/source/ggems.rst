GGEMS
=====


.. sectionauthor:: Julien Bert
.. codeauthor:: Julien Bert


Simulation structure
--------------------

The structure of the Monte Carlo simulation is based on three main components: a source, a phantom and a detector. According the application targeted a composition of these components will be used (see figure below). For instance to compute a blank image in CT imaging only a source and a detector will be used (no phantom). In PET and CT imaging a source, a phantom and a detecvtor will be defined. In particle therapy application only a source and a phantom are used. In GGEMS simulation only the source component is mandatory, phantom and detector are optionals.

.. image:: images/simu_struct.png
    :scale: 40%
    :align: center

The main code of GGEMS application start by defining the different components by instancing their associated c++ classes and setting the different parameters required. Then the Monte Carlo GGEMS engine is instancing as well. Parameters of the simulation (number of particles, physics list, etc.) is selected. Every components are passed to GGEMS. Subsequently GGEMS engine is initialyzed and started. Finally, if a detector was defined results data have to be exported. For more informations about available component in GGEMS see :ref:`sources-label`, :ref:`phantoms-label`, :ref:`detectors-label`. The main c++ classe that handle the complete simulation is ``GGEMS``. This classe allows to setting up the physics list, the simulation parameters, verbosity, etc.

Example:
^^^^^^^^
.. code-block:: cpp
    :linenos:

    // Defined a source
    ConeBeamCTSource *aSource = new ConeBeamCTSource;
    // Set all parameters...

    // Defined a phantom
    VoxPhanImgNav *aPhantom = new VoxPhanImgNav;
    // Set all parameters...

    // Defined a detector
    CTDetector *aDetector = new CTDetector;
    // Set all parameters...

    // Defined a GGEMS simulation
    GGEMS *simu = new GGEMS;
    // Set simulations parameters...

    // Passed every component
    simu->set_source( aSource );
    simu->set_phantom( aPhantom );
    simu->set_detector( aDetector );

    // Init the simulation
    simu->init_simulation();

    // Start the simulation
    simu->start_simulation();

    // Get back the results
    aDetector->save_projection( outputFilename );

-----

New simulation
--------------

A simulation using GGEMS needs to first include the ggems header in source code:

.. code-block:: cpp
    :linenos:

    #include <ggems.cuh>

Each component of the simulation has to be instanciated and setting up (source, phantom and detector). Subsequently a new GGEMS object has to be instanciated:

.. code-block:: cpp
    :linenos:

    // A new GGEMS simulation
    GGEMS *simu = new GGEMS

A GGEMS simulation requires a license data file in order to be executed. This license can be freely obtained for academic institution. This file has to be passed to the simulation:

.. code-block:: cpp
    :linenos:

    // License
    simu->set_license( "license/YOUR_LICENSE_FILE.dat" );


Device architecture
-------------------

GGEMS lets the possiblity to run simulation in both CPU or GPU architecture.

-----

.. c:function:: void set_hardware_target( std::string name )

    Set the hardware to run the simulation

    .. c:var:: name

        Hardware name, can be ``"CPU"`` or ``"GPU"``. CPU means that the complete simulation will be executed using one core of the CPU. GPU means that a major part of the simulation will be executed using one GPU. Default value is ``"GPU"``.


-----

.. c:function:: void set_GPU_block_size( ui32 size )

    Set the size in number of threads of each GPU block. Blocks are used to split data that have to be processed by the GPU. This size depend of your architecture, and not really change the speed of your simulation.

    .. c:var:: size

        Number of threads per block. Default value is 192.

-----

.. c:function:: void set_GPU_ID( ui32 id )

    In case of multiple GPUs or multiple graphic cards you need to select which one you want to use for running the simulation.

    .. c:var:: id

        Number of your GPU. Default value is 0, meaning the first (or the unique) GPU find on your system.


Physics processes
-----------------

Physics processes from GGEMS were originally extracted from the well validated physics of `Geant4 <http://geant4.web.cern.ch/geant4/>`_ (9.4). We try as possible to update processes according each release of Geant4. So far GGEMS include this physics list:

+-----------------------------------------------+-----------------------------------------------+       
| Photons                                       | Electrons                                     |  
+=======================+=======================+=======================+=======================+ 
| Process               | Model                 | Process               | Model                 | 
+-----------------------+-----------------------+-----------------------+-----------------------+ 
| Compton scattering    | standard model        | Ionisation            | Moller Bhabha model   | 
+-----------------------+-----------------------+-----------------------+-----------------------+ 
| Rayleigh scattering   | Livermore model       | Multiple scattering   | Urban93 model         | 
+-----------------------+-----------------------+-----------------------+-----------------------+ 
| Photoelectric effects | standard model        | Bremsstrahlung        | standard model        | 
+-----------------------+-----------------------+-----------------------+-----------------------+ 

-----

.. c:function:: void set_process( std::string proc_name )

    Command that activated a given physics process.

    .. c:var:: proc_name

        Name of the process, can be: ``"Compton"``, ``"Rayleigh"``, ``"PhotoElectric"``, ``"eIonisation"``, ``"eBremsstrahlung"``, ``"eMultipleScattering"``. By default all processes are disabled.

-----

.. c:function:: void set_CS_table_nbins( ui32 nb_bins )

    Cross sections and dE/dx tables are precalculated as Geant4. A number of bins is then required for this stage.

    .. c:var:: nb_bins

        Number of bins of physics table. This number is chosen according your application. In medical application a number of 220 bins is enough. Default value is 220.

-----

.. c:function:: void set_CS_table_E_min( f32 E_min )

    Cross sections and dE/dx tables are precalculated as Geant4. Tables must start by a minimum energy value.

    .. c:var:: E_min

        Minimum energy in MeV used in physics tables. Default value is 990 eV.

-----

.. c:function:: void set_CS_table_E_max( f32 E_max )

    Cross sections and dE/dx tables are precalculated as Geant4. Tables must end by a maximum energy value.

    .. c:var:: E_max

        Maximum energy in MeV used in physics tables. Default value is 250 MeV.

-----

.. c:function:: void set_particle_cut( std::string particle_name, f32 range )

    Energy cut can be applied on particle tracking, this command allows to define this cut in range (mm). GGEMS will calculated for each material the corresponding energy cut.

    .. c:var:: particle_name

        Particle name, can be: ``"Photon"`` or ``"Electron"``.

    .. c:var:: range

        Range cut in mm.

-----

.. c:function:: void set_secondary( std::string particle_name )

    GGEMS can handle secondary particles. To activate their tracking this command is used.

    .. c:var:: particle_name

        Particle name, can be: ``"Photon"`` or ``"Electron"``. By default no secondary particles are tracked. 

.. note::
    Secondary photon particles are not tracked yet. This means that Bremsstrahlung process dropped photon energy locally.


-----

.. c:function:: void set_secondaries_level( ui32 level )

    During simulation secondary particle are buffering to be process. This require a particle queue. The size of this queue is defined by the `level` of secondary particle i.e. the cascade size.

    .. c:var:: level

        Maximum level (size) of the secondary particles queue. Default value is 0 (no secondary). For instance a photon beam of 1 MeV (without energy cut) a level of 6 is necessary. Meanning that secondary particle will created other secondary particles and so on, until 6 level of secondaries. This parameters impact on the memory size require to store particles within the graphic card.

-----

.. c:function:: void set_geometry_tolerance( f32 range )

    Particle transportation is calculated using raytracing function that require some tolerance value to consider boundary approximation between object. This command allows to change this tolerance.

    .. c:var:: range

        Range tolerance in mm. Default value is 100 nm.

.. warning::
    This function must be used by expert user. Any inappropriate value will have dramatically change on the simulation.


Execution
---------

.. c:function:: void set_source ( GGEMSSource* aSource )

    Assign a source to the simulation. This function is mandatory, meaning that at least a source has to be defined in GGEMS.

    .. c:var:: aSource

        Source component from the ones available in GGEMS ( :ref:`sources-label` ). Default value is ``NULL``.

-----

.. c:function:: void set_phantom ( GGEMSPhantom* aPhantom )

    Assign a phantom to the simulation.

    .. c:var:: aPhantom

        Phantom component from the ones available in GGEMS ( :ref:`phantoms-label` ). Default value is ``NULL``.

-----

.. c:function:: void set_detector( GGEMSDetector* aDetector )

    Assign a detector to the simulation.

    .. c:var:: aDetector

        Detector component from the ones available in GGEMS ( :ref:`detectors-label` ). Default value is ``NULL``.

-----

.. c:function:: void set_seed ( ui32 seed )

    Select a seed to initialize the pseudo random number generator.

    .. c:var:: seed

        Seed value. Default value is 0, meaning that a random seed is calculated.

-----

.. c:function:: void set_number_of_particles ( ui64 nb )

    Choose the total number of particles required by the simulation.

    .. c:var:: nb

        Total number of particles.

-----

.. c:function:: void set_size_of_particles_batch ( ui64 nb )

    GGEMS proceed simulation by tracking batch of particle. This command allows to chosse the size of each batch.

    .. c:var:: nb

        Number of particles per batch. This number must be chosen accordingly to the global memeory available on the graphic card. For example a :math:`10^6` particles batch size with 4 levels of secondary particle required 204 MB of memory.


-----

.. c:function:: void init_simulation()

    Once everythings has been set up, GGEMS simulation must be initialized. GGEMS will pre-calculate physics table, materials table and load data into GPU memory.

-----

.. c:function:: void start_simulation()

    Start GGEMS simulation. Before running you need to initialized the simulation.


Verbosity
---------

Verbosity is usefull to know the different parameters of the simulation.

-----

.. c:function:: set_display_in_color( bool value )

    Verbosity is in color within linux terminal, however for Windows user or for storing log file, it is possible to switch off color verbosity.

    .. c:var:: value

        Flag value, can be ``false`` or ``true``. By default this value is ``true`` except for Windows user, where the flag is force to ``false``.

-----

.. c:function:: set_display_memory_usage( bool value )

    This command allows estimating the memory requires by each component (source, phantom and detector). This is usefull to know if everything will fit into graphic card memory.

    .. c:var:: value

        Flag value, can be ``false`` or ``true``. By default this value is ``false``.

-----

.. c:function:: set_display_energy_cuts( bool value )

    Each energy cut (photon and electron) for each material will be displayed.

    .. c:var:: value

        Flag value, can be ``false`` or ``true``. By default this value is ``false``.

Example
-------

.. code-block:: cpp
    :linenos:

    // GGEMS simulation
    GGEMS *simu = new GGEMS;

    // Licence
    simu->set_license( "license/YOUR_LICENSE_FILE.dat" );

    // GPU parameters
    simu->set_hardware_target( "GPU" );
    simu->set_GPU_block_size( 192 );
    simu->set_GPU_ID( 0 );

    // Physics parameters
    simu->set_process( "Compton" );
    simu->set_process( "PhotoElectric" );
    simu->set_process( "Rayleigh" );
    
    simu->set_process( "eIonisation" );
    simu->set_process( "eBremsstrahlung" );
    simu->set_process( "eMultipleScattering" );

    simu->set_secondaries_level( 6 );
    simu->set_secondary( "Electron" );

    // Energy table range
    simu->set_CS_table_nbins( 220 );
    simu->set_CS_table_E_min( 990.*eV );
    simu->set_CS_table_E_max( 250.*MeV );

    simu->set_particle_cut( "photon", 100 *um );   
    simu->set_particle_cut( "electron", 100 *um ); 

    // Random and particles
    simu->set_seed( 123456789 );
    simu->set_number_of_particles( 1000000 );
    simu->set_size_of_particles_batch( 100000 );
    
    // Source and phantom
    simu->set_source( aSource );
    simu->set_phantom( aPhantom );

    // Verbose
    simu->set_display_in_color( true );    
    simu->set_display_memory_usage( true );

    // Initialization of the simulation
    simu->init_simulation();

    // Start the simulation
    simu->start_simulation();



Last update: |today|  -  Release: |release|.