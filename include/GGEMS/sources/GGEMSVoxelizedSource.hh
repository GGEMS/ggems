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
  \date Monday September 15, 2025
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

  private:
    GGbool is_monoenergy_mode_; /*!< Boolean checking the mode of energy */
    GGfloat monoenergy_; /*!< Monoenergy mode */
    std::string energy_spectrum_filename_; /*!< The energy spectrum filename for polyenergetic mode */
    GGsize number_of_energy_bins_; /*!< Number of energy bins for the polyenergetic mode */
    cl::Buffer** energy_spectrum_; /*!< Energy spectrum for OpenCL device */
    cl::Buffer** energy_cdf_; /*!< Cumulative distribution function for energy to generate a random energy */
};

/*!
  \fn GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name)
  \return the pointer on the singleton
  \param source_name - name of the source
  \brief Get the GGEMSVoxelizedSource pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSVoxelizedSource* create_ggems_voxelized_source(char const* source_name);

#endif // End of GUARD_GGEMS_SOURCES_GGEMSVOXELIZEDSOURCE_HH
