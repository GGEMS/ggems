// GGEMS Copyright (C) 2015

/*!
 * \file beamlet_source.cuh
 * \brief Beamlet source
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Thursday May 19, 2016
 *
 */

#ifndef BEAMLET_SOURCE_CUH
#define BEAMLET_SOURCE_CUH

#include "global.cuh"
#include "particles.cuh"
#include "ggems_source.cuh"
#include "prng.cuh"
#include "vector.cuh"
#include "fun.cuh"

class GGEMSource;

/*!
  \class BeamletSource
  \brief This class is a source that produces beamlet particles source.
*/
class BeamletSource : public GGEMSSource
{
public:
    /*!
     * \brief BeamletSource contructor
     */
    BeamletSource();

    /*!
     * \brief BeamletSource destructor
     */
    ~BeamletSource();

    /*!
     * \fn void set_position_in_beamlet_plane( f32 posx, f32 posy )
     * \brief Set the position of the beamlet (local x-y plane, isocenter is the ref)
     * \param posx Position of the source in X
     * \param posy Position of the source in Y
     */
    void set_position_in_beamlet_plane( f32 posx, f32 posy );

    /*!
     * \fn void set_distance_to_isocenter( f32 dis )
     * \brief Set the distance of the beamlet x-y plane to the isocenter
     * \param dis Distance
     */
    void set_distance_to_isocenter( f32 dis );

    /*!
     * \fn void set_rotation( f32 agx, f32 agy, f32 agz )
     * \brief Set the orientation of the beamlet
     * \param agx Angle along x-axis (non-coplanar angle)
     * \param agy Angle along y-axis (Carousel rotation)
     * \param agz Angle along z-axis (Gantry angle)
     */
    void set_rotation( f32 agx, f32 agy, f32 agz );

    /*!
     * \fn void set_particle_type( std::string pname )
     * \brief Set the particle type
     * \param pname Type of the particle ("photon", "electron", etc)
     */
    void set_particle_type( std::string pname );

    /*!
      \fn void set_mono_energy( f32 energy )
      \param energy Monoenergy value
      \brief Set the energy value
    */
    void set_mono_energy( f32 energy );

    /*!
      \fn void set_energy_spectrum( std::string filename )
      \param filename Filename of the polychromatic source file
      \brief Set the spectrum file of the beamlet
    */
    void set_energy_spectrum( std::string filename );

    /*!
     * \fn void set_focal_point( f32 posx, f32 posy, f32 posz );
     * \brief Set the focal point of the emission LINAC source
     * \param posx Position of the source in X
     * \param posy Position of the source in Y
     * \param posz Position of the source in Z
     */
    void set_focal_point( f32 posx, f32 posy, f32 posz );

    /*!
     * \fn void set_size( f32 sizex, f32 sizey )
     * \brief Set the size of the beamlet
     * \param sizex Beamlet size along x-axis
     * \param sizey Beamlet size along y-axis
     */
    void set_size( f32 sizex, f32 sizey );

public: // Abstract from GGEMSSource (Mandatory funtions)

    /*!
     * \fn void get_primaries_generator( Particles particles )
     * \brief Generate particles
     * \param particles Stack of particles
     */
    void get_primaries_generator( Particles particles );

    /*!
     * \brief Initialize the source before running the simualtion
     * \param params Simulations parameters
     */
    void initialize( GlobalSimulationParameters params );

private: // Make BeamletSource class non-copyable
    /*!
    \brief Copy constructor not implement and impossible to use for the user by security
  */
    BeamletSource( BeamletSource const& );

    /*!
    \brief Copy assignment not implement and impossible to use for the user by security
  */
    BeamletSource& operator=( BeamletSource const& );

    /*!
   * \fn void m_load_spectrum()
   * \brief Function that reads and loads a spectrum in memory
   */
    void m_load_spectrum();

private:    
    f32xy m_pos;                          /*!< Position of the beamlet in 2D (x, y, z) */
    f32 m_dist;                           /*!< Distance to the isocenter */
    f32xyz m_foc_pos;                     /*!< Position of the beamelt focal in 3D (x, y, z) */
    f32xyz m_angle;                       /*!< Orientation of the beamlet */
    f32xy m_size;                         /*!< Beamlet size in 2D (x, y) */
    ui8 m_particle_type;                  /*!< Type of the particle */
    std::string m_spectrum_filename;      /*!< Name of the file that contains the spectrum */
    f32 *m_spectrum_E;                    /*!< Energy spectrum of the source on the host (CPU) */
    f32 *m_spectrum_CDF;                  /*!< CDF of the source on the host (CPU) */
    f32 m_energy;                         /*!< In case of mono energy, the energy value */
    ui32 m_nb_of_energy_bins;             /*!< Number of the bins in the energy spectrum */
    f32matrix44 m_transform;              /*!< Trsnformation matrix */
    GlobalSimulationParameters m_params;  /*!< Simulation parameters */
};

#endif
