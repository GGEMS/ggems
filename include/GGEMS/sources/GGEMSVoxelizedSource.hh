#ifndef GUARD_GGEMS_SOURCES_GGEMSVOXELIZEDSOURCE_HH
#define GUARD_GGEMS_SOURCES_GGEMSVOXELIZEDSOURCE_HH

// ************************************************************************
// * This file is part of GGEMS.                                          *
// *                                                                      *
// * GGEMS is free software: you can redistribute it and/or modify        *
// * it under the terms of the GNU General Public License as published by *
// * the Free Software Foundation, either version 3 of the License, or    *
// * (at your option) any later version.                                  *
// *                                                                      *
// * GGEMS is distributed in the hope that it will be useful,             *
// * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
// * GNU General Public License for more details.                         *
// *                                                                      *
// * You should have received a copy of the GNU General Public License    *
// * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
// *                                                                      *
// ************************************************************************

/*!
  \file GGEMSVoxelizedSource.hh

  \brief This class defines a voxelized source in GGEMS useful for SPECT/PET simulations

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 1, 2025
*/

#include "GGEMS/sources/GGEMSSource.hh"

/*!
  \class GGEMSVoxelizedSource
  \brief This class defines a voxelized source in GGEMS useful for SPECT/PET simulations
*/
class GGEMS_EXPORT GGEMSVoxelizedSource : public GGEMSSource
{
  public:
    /*!
      \param source_name - name of the source
      \brief GGEMSVoxelizedSource constructor
    */
    explicit GGEMSVoxelizedSource(std::string const& source_name);

    /*!
      \brief GGEMSVoxelizedSource destructor
    */
    ~GGEMSVoxelizedSource(void) override;

    /*!
      \fn GGEMSVoxelizedSource(GGEMSVoxelizedSource const& voxelized_source) = delete
      \param voxelized_source - reference on the GGEMS voxelized source
      \brief Avoid copy by reference
    */
    GGEMSVoxelizedSource(GGEMSVoxelizedSource const& voxelized_source) = delete;

    /*!
      \fn GGEMSVoxelizedSource& operator=(GGEMSVoxelizedSource const& voxelized_source) = delete
      \param voxelized_source - reference on the GGEMS voxelized source
      \brief Avoid assignement by reference
    */
    GGEMSVoxelizedSource& operator=(GGEMSVoxelizedSource const& voxelized_source) = delete;

    /*!
      \fn GGEMSVoxelizedSource(GGEMSVoxelizedSource const&& voxelized_source) = delete
      \param voxelized_source - rvalue reference on the GGEMS voxelized source
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedSource(GGEMSVoxelizedSource const&& voxelized_source) = delete;

    /*!
      \fn GGEMSVoxelizedSource& operator=(GGEMSVoxelizedSource const&& voxelized_source) = delete
      \param voxelized_source - rvalue reference on the GGEMS voxelized source
      \brief Avoid copy by rvalue reference
    */
    GGEMSVoxelizedSource& operator=(GGEMSVoxelizedSource const&& voxelized_source) = delete;

    /*!
      \fn void Initialize(bool const& is_tracking = false)
      \param is_tracking - flag activating tracking
      \brief Initialize a GGEMS source
    */
    void Initialize(bool const& is_tracking = false) override;

    /*!
      \fn void PrintInfos(void) const
      \brief Printing infos about the source
    */
    void PrintInfos(void) const override;

    /*!
      \fn void GetPrimaries(GGsize const& thread_index, GGsize const& number_of particles)
      \param thread_index - index of activated device (thread index)
      \param number_of_particles - number of particles to generate
      \brief Generate primary particles
    */
    void GetPrimaries(GGsize const& thread_index, GGsize const& number_of_particles) override;

    /*!
      \fn void SetPhantomSourceFile(std::string const& phantom_source_filename)
      \param phantom_source_filename - phantom source file for vox. source
      \brief Set phantom source file in MHD format for voxelized source
    */
    void SetPhantomSourceFile(std::string const& phantom_source_filename);

  private:
    /*!
      \fn void InitializeKernel(void)
      \brief Initialize kernel for specific source in OpenCL
    */
    void InitializeKernel(void) override;

    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a source
    */
    void CheckParameters(void) const override;

    /*!
      \fn void FillVoxelActivity(void)
      \brief fill activity source for phantom file to compute cdf
    */
    void FillVoxelActivity(void);

  private:
    std::string phantom_source_filename_; /*!< The phantom source filename for voxelized source */
    cl::Buffer** phantom_vox_data_; /*!< Data about voxelized source */
    GGint number_of_activity_bins_; /*!< Number of bins for cdf activity with non zero activity */
    cl::Buffer** activity_index_; /*!< Activity index of voxel */
    cl::Buffer** activity_cdf_; /*!< Cumulative distribution function to generate a particle in a certain voxel */
};

/*!
  \fn GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name)
  \return the pointer on the singleton
  \param source_name - name of the source
  \brief Get the GGEMSVoxelizedSource pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name);

/*!
  \fn void set_position_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit)
  \param voxelized_source - pointer on the source
  \param pos_x - Position of the source in X
  \param pos_y - Position of the source in Y
  \param pos_z - Position of the source in Z
  \param unit - unit of the distance
  \brief Set the position of the source in the global coordinates
*/
extern "C" GGEMS_EXPORT void set_position_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const pos_x, GGfloat const pos_y, GGfloat const pos_z, char const* unit);

/*!
  \fn void set_number_of_particles_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGsize const number_of_particles)
  \param voxelized_source - pointer on the source
  \param number_of_particles - number of particles to simulate
  \brief Set the number of particles to simulate during the simulation
*/
extern "C" GGEMS_EXPORT void set_number_of_particles_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGsize const number_of_particles);

/*!
  \fn void set_source_particle_type_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* particle_name)
  \param voxelized_source - pointer on the source
  \param particle_name - name/type of the particle: photon or electron
  \brief Set the type of the source particle
*/
extern "C" GGEMS_EXPORT void set_source_particle_type_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* particle_name);

/*!
  \fn void set_monoenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const monoenergy, char const* unit)
  \param voxelized_source - pointer on the source
  \param monoenergy - monoenergetic value
  \param unit - unit of the energy
  \brief Set the monoenergy value for the GGEMSVoxelizedSource
*/
extern "C" GGEMS_EXPORT void set_monoenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const monoenergy, char const* unit);

/*!
  \fn void set_polyenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* energy_spectrum)
  \param voxelized_source - pointer on the source
  \param energy_spectrum - polyenergetic spectrum
  \brief Set the polyenergetic spectrum value for the GGEMSVoxelizedSource
*/
extern "C" GGEMS_EXPORT void set_polyenergy_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* energy_spectrum);

/*!
  \fn void set_phantom_source_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* phantom_source_file)
  \param voxelized_source - pointer on the source
  \param phantom_source_file - phantom source file
  \brief Set the phantom source file for vox. source
*/
extern "C" GGEMS_EXPORT void set_phantom_source_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, char const* phantom_source_file);

/*!
  \fn void set_energy_peak_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const energy, char const* unit, GGfloat const intensity)
  \param voxelized_source - pointer on the source
  \param energy - energy value
  \param unit - unit of the energy
  \param intensity - intensity of bin
  \brief Set the energy peak spectrum value for the GGEMSVoxelizedSource
*/
extern "C" GGEMS_EXPORT void set_energy_peak_ggems_voxelized_source(GGEMSVoxelizedSource* voxelized_source, GGfloat const energy, char const* unit, GGfloat const intensity);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSVOXELIZEDSOURCE_HH
