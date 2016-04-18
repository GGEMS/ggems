.. GGEMS documentation: Detectors

.. _detectors-label:

Detectors
=========

CTDetector
----------

.. sectionauthor:: Julien Bert
.. codeauthor:: Didier Benoit

This detector in mainly used in CT applications. Flatpanel can orbit around the patient and count the number of particles reaching the detector using an energy threshold. Particles are not navigated within the detector.

------------

.. c:function:: void set_dimension( f32 width, f32 height, f32 depth )
    
    Set dimension of the detector.

    .. c:var:: width  
        
        Width of the detector (in mm).

    .. c:var:: height 
    
        Height of the detector (in mm).
        
    .. c:var:: depth 
    
        Depth of the detector (in mm).

------------

.. c:function:: void set_pixel_size( f32 sx, f32 sy, f32 sz )
    
    Set the pixel size of the detector.

    .. c:var:: sx  
        
        Pixel size along x-axis (in mm).

    .. c:var:: sy 
    
        Pixel size along y-axis (in mm).
        
    .. c:var:: sz 
    
        Pixel size along z-axis (in mm).

------------

.. c:function:: void set_position( f32 px, f32 py, f32 pz )
    
    Set the detector's position in world space.

    .. c:var:: px  
        
        Position along x-axis (in mm).

    .. c:var:: py 
    
        Position along y-axis (in mm).
        
    .. c:var:: pz 
    
        Position along z-axis (in mm).


------------

.. c:function:: void set_threshold( f32 threshold )
    
    Define an energy threshold to detect particles.

    .. c:var:: threshold  
        
        Energy threshold in MeV. Below this value, particles are not detected.

------------

.. c:function:: void set_orbiting( f32 orbiting_angle )
    
    Set the orbiting angle of the detector.

    .. c:var:: orbiting_angle  
        
        Angle in degrees.

-----

.. c:function:: void save_projection( std::string filename )
    
    Save projection recovered by the detector to a MetaImage file.

    .. c:var:: filename  
        
        Filename of the projection.

-----

.. c:function:: void save_scatter( std::string filename )
    
    Save scatter projection recovered by the detector to a MetaImage file.

    .. c:var:: filename  
        
        Filename of the scatter projection.


.. note::
    Version: beta - work for authors.

Example
^^^^^^^

.. code-block:: cpp
    :linenos:

    // Defined a detector
    CTDetector *aDetector = new CTDetector;    
    aDetector->set_dimension( 1, 780, 710 ); // in pixel
    aDetector->set_pixel_size( 0.600f*mm, 0.368f*mm, 0.368f*mm );
    aDetector->set_position( 320.3f*mm, 0.0f*mm, 0.0f*mm );
    aDetector->set_threshold( 10.0f*keV );
    aDetector->set_orbiting( 3.6f*deg );  // same angle from the source



Last update: |today|  -  Release: |release|.
