.. GGEMS documentation: Sources

.. _sources-label:

Sources
=======

ConeBeamCTSource
----------------

.. sectionauthor:: Julien Bert
.. codeauthor:: Didier Benoit

Class cone-beam source. The user can define a focal, an aperture, an orbiting angle, etc. This source mainly used in CT application.

------------

.. c:function:: void set_position( f32 posx, f32 posy, f32 posz)
    
    Set the position of the center of the source.

    .. c:var:: posx  
        
        Position of the source in along the x-axis (in mm).

    .. c:var:: posy 
    
        Position of the source in along the y-axis (in mm).
        
    .. c:var:: posz 
    
        Position of the source in along the z-axis (in mm).


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
        
        Aperture in radian of the X-ray CT source.

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
    must have two colums. A first one that list energy and a second one that list probability of the spectrum.

    .. c:var:: filename

        Filename of the polychromatic source file.

------------

.. c:function:: void set_orbiting( f32 orbiting_angle )

    Rotate the source around the z-axis and based on the center of the system.

    .. c:var:: orbiting_angle

        Orbiting angle around the center of the system in radian.


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






Last update: |today|  -  Release: |release|.
