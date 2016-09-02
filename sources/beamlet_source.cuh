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
     * \fn void set_beamlet_relative_position( f32 posx, f32 posy, f32 posz )
     * \brief Set the position of the beamlet (local position within the beamlet plane)
     * \param posx Position of the source in X
     * \param posy Position of the source in Y
     * \param posz Position of the source in Z
     */
    void set_local_beamlet_position( f32 posx, f32 posy, f32 posz );

    /*!
     * \fn void set_beamlet_origin( f32 posx, f32 posy, f32 posz )
     * \brief Set the global origin position of the beamlet plane (from the isocenter)
     * \param posx Position of the origin in X
     * \param posy Position of the origin in Y
     * \param posz Position of the origin in Z
     */
    void set_frame_position( f32 posx, f32 posy, f32 posz );

    /*!
     * \fn void set_local_source_position( f32 posx, f32 posy, f32 posz )
     * \brief Set the origin of the LINAC source
     * \param posx Position of the origin in X
     * \param posy Position of the origin in Y
     * \param posz Position of the origin in Z
     */
    void set_local_source_position( f32 posx, f32 posy, f32 posz );

    /*!
     * \fn void set_frame_axis( f32 m00, f32 m01, f32 m02, f32 m10, f32 m11, f32 m12, f32 m20, f32 m21, f32 m22 )
     * \brief Set the axis transformation of the beamlet plane compare to the global frame
     * \param m00 Element of the matrix
     * \param m01 Element of the matrix
     * \param m02 Element of the matrix
     * \param m10 Element of the matrix
     * \param m11 Element of the matrix
     * \param m12 Element of the matrix
     * \param m20 Element of the matrix
     * \param m21 Element of the matrix
     * \param m22 Element of the matrix
     */
    void set_frame_axis( f32 m00, f32 m01, f32 m02,
                         f32 m10, f32 m11, f32 m12,
                         f32 m20, f32 m21, f32 m22 );

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
     * \fn void set_mono_energy( f32 energy )
     * \param energy Monoenergy value
     * \brief Set the energy value
     */
    void set_mono_energy( f32 energy );

    /*!
     * \fn void set_energy_spectrum( std::string filename )
     * \param filename Filename of the polychromatic source file
     * \brief Set the spectrum file of the beamlet
     */
    void set_energy_spectrum( std::string filename );

    /*!
     * \fn void set_size( f32 sizex, f32 sizey )
     * \brief Set the size of the beamlet
     * \param sizex Beamlet size along x-axis
     * \param sizey Beamlet size along y-axis
     */
    void set_local_size( f32 sizex, f32 sizey, f32 sizez );

public:
    f32xyz get_local_beamlet_position();
    f32xyz get_local_source_position();
    f32xyz get_local_size();
    f32matrix44 get_transformation_matrix();

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
    f32xyz m_pos;                         /*!< Position of the beamlet in 3D (x, y, z) */
    f32xyz m_org;                         /*!< Origin of the beamlet plane in 3D (x, y, z) */
    f32xyz m_src;                         /*!< Origin of the LINAC source */
    f32matrix33 m_axis_trans;             /*!< Axis transformation matrix */
    f32xyz m_angle;                       /*!< Orientation of the beamlet */
    f32xyz m_size;                        /*!< Beamlet size in 3D (x, y, z) */
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
