.. GGEMS documentation: Sources

.. _sources-label:

Sources
=======

ConeBeamCTSource
----------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Didier Benoit

Class cone-beam source. The user can define a focal, an aperture, an orbiting angle, etc. This source is mainly used in CT applications.

------------

.. c:function:: void set_position( f32 posx, f32 posy, f32 posz)
    
    Set the position of the center of the source.

    .. c:var:: posx  
        
        Position of the source along the x-axis (in mm).

    .. c:var:: posy 
    
        Position of the source along the y-axis (in mm).
        
    .. c:var:: posz 
    
        Position of the source along the z-axis (in mm).


------------

.. c:function:: void set_focal_size( f32 hfoc, f32 vfoc )

    Set the focal size of the cone-beam source.

    .. c:var:: hfoc  
        
        Horizontal focal of the source (in mm).

    .. c:var:: vfoc 
    
        Vertical focal of the source (in mm).

------------

.. c:function:: void set_beam_aperture( f32 aperture )

    Set the aperture of the source.

    .. c:var:: aperture  
        
        Aperture of the X-ray CT source in radians.

------------

.. c:function:: void set_particle_type( std::string pname )

    Set the type of the particle.

    .. c:var:: pname

        Name of the particle ("photon" or "electron").

------------

.. c:function:: void set_mono_energy( f32 energy )

    Set the energy value of the particles. All particles will get the same energy.

    .. c:var:: energy

        Monoenergy value in MeV.

------------

.. c:function:: void set_energy_spectrum( std::string filename )

    Set the spectrum of the source based on a histogram file. This file in text format
    must have two columns. A first one listing the energy and a second one the probability of the spectrum.

    .. c:var:: filename

        Filename of the polychromatic source file.

------------

.. c:function:: void set_orbiting( f32 orbiting_angle )

    Rotate the source around the z-axis with respect to the center of the system.

    .. c:var:: orbiting_angle

        Orbiting angle around the center of the system in radians.


.. note::
    Version: beta - work for authors.

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    ConeBeamCTSource *aSource = new ConeBeamCTSource;
    aSource->set_position( 950*mm, 0.0*mm, 0.0*mm );
    aSource->set_orbiting( 12.0*deg );
    aSource->set_particle_type( "photon" );
    aSource->set_focal_size( 0.6*mm, 1.2*mm );
    aSource->set_beam_aperture( 8.7*deg );
    aSource->set_energy_spectrum( "data/spectrum_120kVp_2mmAl.dat" );


PhaseSpaceSource
----------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Didier Benoit

Class phase-space source. This source allows the use of IAEA phase-space file or MHD phase-space file ( :ref:`mhd-label` ). Phase-space can be duplicated and transformed to simulate multiple virtual sources.

------------

.. c:function:: void set_translation( f32 tx, f32 ty, f32 tz)
    
    If only one source is required the position of the phase-space can be translate using
    this function.

    .. c:var:: tx  
        
        Translation along the x-axis (in mm).

    .. c:var:: ty 
    
        Translation along the y-axis (in mm).
        
    .. c:var:: tz 
    
        Translation along the z-axis (in mm).

-----

.. c:function:: void set_rotation( f32 rx, f32 ry, f32 rz)
    
    If only one source is required the phase-space can be rotate using
    this function. Yaw, pitch, and roll convention is used with the right-hand rule.

    .. c:var:: rx  
        
        Rotation around the x-axis (in rad).

    .. c:var:: ry 
    
        Rotation around the y-axis (in rad).
        
    .. c:var:: rz 
    
        Rotation around the z-axis (in rad).

-----

.. c:function:: void load_phasespace_file( std::string filename )
    
    Load a phase-space file in IAEA format or MHD format ( :ref:`mhd-label` ). Before load, please check if you have enough memory
    on your graphics card.

    .. c:var:: filename  
        
        Header name of the IAEA phase-space file or mhd phase-space file.

-----

.. c:function:: void load_transformation_file( std::string filename )
    
    In case of multiple virtual sources, a file containing every source transformation can be used.
    For each virtual source, translation, rotation, scaling and activity (i.e. emission probability) has
    to be specified.

    .. c:var:: filename  
        
        Text file with every transformation.

Transformation file must absolutely follows this format::

    # File that duplicate and transform the phasespace
    # Translations are in mm and rotation in degrees
    # (mm)      (degree)  (0.0 to 1.0)  (Arbitrary Unit i.e. probabilty of emission)
    # tx ty tz  rx ry rz  sx sy sz      activity
    # example with two sources
    -20.0 0.0 0.0  0.0 0.0 0.0 1.0 1.0 1.0 0.5
     20.0 0.0 0.0 30.0 0.0 0.0 1.0 1.0 1.0 1.0

----

.. c:function:: void set_max_number_of_particles( ui32 nb_of_particles )

    Select the maximum number of particles used within the phase-space. This allow to virtually resize the phase-space by truncating the number of particles contained.

    .. c:var:: nb_of_particles

        Number of particles, default value is -1, meaning that all particles contain within the phase-space will be used.

------------

.. note::
    Version: beta - work for authors.

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    PhaseSpaceSource *aSource = new PhaseSpaceSource;    
    aSource->load_phasespace_file( "data/output.IAEAheader" );
    aSource->load_transformation_file( "data/transformation.dat" );

Last update: |today|  -  Release: |release|.
