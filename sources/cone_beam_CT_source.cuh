#ifndef GUARD_CONE_BEAM_CT_SOURCE_CUH
#define GUARD_CONE_BEAM_CT_SOURCE_CUH

// GGEMS Copyright (C) 2015

/*!
 * \file cone_beam_CT_source.cuh
 * \brief Cone beam source for CT
 * \author Didier Benoit <didier.benoit13@gmail.com>
 * \version 0.1
 * \date Friday January 8, 2015
*/

#include "global.cuh"
#include "particles.cuh"

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
      \fn void set_focal_size( f32 hfoc, f32 vfoc )
      \param hfoc horizontal focal of the source
      \param vfoc vertical focal of the source
      \brief Set the focal size of the cone-beam source
    */
    void set_focal_size( f32 hfoc, f32 vfoc );

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
      \fn void set_direction ( std::string type, f32 vdx, f32 vdy, f32 vdz )
      \param type Type of the source
      \param vdx Direction of the X-ray beam in X
      \param vdy Direction of the X-ray beam in Y
      \param vdz Direction of the X-ray beam in Z
      \brief Set the direction of the X-ray beam
    */
    void set_direction( std::string type, f32 vdx, f32 vdy, f32 vdz );

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

    /*!
      \fn void set_orbiting( f32 orbiting_angle )
      \param orbiting_angle orbiting angle around the center of the system
      \brief Rotate the source around the center of the system
    */
    void set_orbiting( f32 orbiting_angle );

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
    void get_primaries_generator( Particles particles );

    /*!
      \fn void initialize( GlobalSimulationParameters params )
      \param params simulation parameters
      \brief initialize the source for the simulation
    */
    void initialize( GlobalSimulationParameters params );

  private: // Make ConeBeamCTSource class non-copyable
    /*!
      \brief Copy constructor not implement and impossible to use for the user by security
    */
    ConeBeamCTSource( ConeBeamCTSource const& );

    /*!
      \brief Copy assignment not implement and impossible to use for the user by security
    */
    ConeBeamCTSource& operator=( ConeBeamCTSource const& );

  private:
    f32 m_px; /*!< Position of the source in X */
    f32 m_py; /*!< Position of the source in Y */
    f32 m_pz; /*!< Position of the source in Z */
    f32 m_hfoc; /*!< Horizontal focal position */
    f32 m_vfoc; /*!< Vertical focal position */
    f32 m_aperture; /*!< Aperture of the source */
    ui8 m_particle_type; /*!< Type of the particle */
    ui8 m_direction_option; /*!< Direction option for the beam */
    f32 m_orbiting_angle; /*!< Orbiting angle of the source */
    f32 m_dx; /*!< Direction in X for the beam */
    f32 m_dy; /*!< Direction in Y for the beam */
    f32 m_dz; /*!< Direction in Z for the beam */
    f64 *m_spectrumE_h; /*!< Energy spectrum of the source on the host (CPU) */
    f64 *m_spectrumE_d; /*!< Energy spectrum of the source on the device (GPU) */
    f64 *m_spectrumCDF_h; /*!< CDF of the source on the host (CPU) */
    f64 *m_spectrumCDF_d; /*!< CDF of the source on the device (GPU) */
    ui32 m_nb_of_energy_bins; /*!< Number of the bins in the energy spectrum */
    GlobalSimulationParameters m_params; /*!< Simulation parameters */
};

#endif

