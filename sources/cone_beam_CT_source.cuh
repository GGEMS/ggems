#ifndef GUARD_CONE_BEAM_CT_SOURCE_CUH
#define GUARD_CONE_BEAM_CT_SOURCE_CUH

// GGEMS Copyright (C) 2017

/*!
 * \file cone_beam_CT_source.cuh
 * \brief Cone beam source for CT
 * \author Didier Benoit <didier.benoit13@gmail.com>
 * \author Julien Bert <bert.jul@gmail.com>
 * \version 0.3
 * \date Friday January 8, 2015
 *
 * v0.3 - JB: Change all structs and remove CPU exec
 * v0.2 - JB: Add local and global frame and unified memory
 * v0.1 - DB: First code
*/

#include "global.cuh"
#include "vector.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"

class GGEMSSource;

/*!
  \class ConeBeamCTSource
  \brief Class cone-beam source for composed by different methods to
  characterize the source. In this source, the user can define a focal, non
  only a point source.
*/
class ConeBeamCTSource : public GGEMSSource
{
public:
    /*!
      \brief ConeBeamCTSource constructor
    */
    ConeBeamCTSource();

    /*!
      \brief ConeBeamCTSource destructor
    */
    ~ConeBeamCTSource();

    /*!
      \fn void set_position( f32 px, f32 py, f32 pz )
      \param px position of the source in X
      \param py position of the source in Y
      \param pz position of the source in Z
      \brief Set the position of the center of the source
    */
    void set_position( f32 px, f32 py, f32 pz );

    /*!
      \fn void set_orbiting( f32 orbiting_angle )
      \param orbiting_angle orbiting angle around the center of the system
      \brief Rotate the source around the center of the system
    */
    void set_rotation( f32 rx, f32 ry, f32 rz );

    void set_local_axis( f32 m00, f32 m01, f32 m02,
                         f32 m10, f32 m11, f32 m12,
                         f32 m20, f32 m21, f32 m22 );

    /*!
      \fn void set_focal_size( f32 hfoc, f32 vfoc )
      \param hfoc horizontal focal of the source
      \param vfoc vertical focal of the source
      \brief Set the focal size of the cone-beam source
    */
    void set_focal_size( f32 xfoc, f32 yfoc, f32 zfoc );

    /*!
      \fn void set_beam_aperture ( f32 aperture )
      \param aperture aperture in degree of the X-ray CT source
      \brief Set the aperture of the source
    */
    void set_beam_aperture( f32 aperture );

    /*!
      \fn void set_particle_type ( std::string pname )
      \param pname name of the particle
      \brief set the type of the particle
    */
    void set_particle_type( std::string pname );

    /*!
      \fn void set_mono_energy( f32 energy )
      \param energy monoenergy value
      \brief set the energy value
    */
    void set_mono_energy( f32 energy );

    /*!
      \fn void set_energy_spectrum( std::string filename )
      \param filename filename of the polychromatic source file
      \brief set the histogram file of the source
    */
    void set_energy_spectrum( std::string filename );

    f32xyz get_position();

    f32xyz get_orbiting_angles();

    f32 get_aperture();

    f32matrix44 get_transformation_matrix();

    /*!
      \fn std::ostream& operator<<( std::ostream& os, ConeBeamCTSource const& cbct )
      \param os output stream
      \param cbct source object
      \brief stream extraction
    */
    friend std::ostream& operator<<( std::ostream& os,
                                     ConeBeamCTSource const& cbct );

public: // Mandatory method from GGEMSSource abstract class

    /*!
      \fn void get_primaries_generator( Particles particles )
      \param particles particle to generate for the simulation
      \brief generation of the particle
    */
    void get_primaries_generator( ParticlesData *d_particles );

    /*!
      \fn void initialize( GlobalSimulationParameters params )
      \param params simulation parameters
      \brief initialize the source for the simulation
    */
    void initialize( GlobalSimulationParametersData *h_params );

private: // Make ConeBeamCTSource class non-copyable
    /*!
      \brief Copy constructor not implement and impossible to use for the user by security
    */
    ConeBeamCTSource( ConeBeamCTSource const& );

    /*!
      \brief Copy assignment not implement and impossible to use for the user by security
    */
    ConeBeamCTSource& operator=( ConeBeamCTSource const& );

    /*!
     * \fn void m_load_spectrum()
     * \brief Function that reads and loads a spectrum in memory
     */
    void m_load_spectrum();

private:

    f32xyz m_pos;

    f32xyz m_foc;

    f32matrix33 m_local_axis;

    f32matrix44 m_transform;

    f32 m_aperture; /*!< Aperture of the source */
    ui8 m_particle_type; /*!< Type of the particle */
    f32xyz m_angles; /*!< Orbiting angle of the source */

    f32 *m_spectrum_E;
    f32 *m_spectrum_CDF;
    f32 m_energy;

    std::string m_spectrum_filename;      /*!< Name of the file that contains the spectrum */

    ui32 m_nb_of_energy_bins; /*!< Number of the bins in the energy spectrum */
    GlobalSimulationParametersData *mh_params; /*!< Simulation parameters */
};

#endif

