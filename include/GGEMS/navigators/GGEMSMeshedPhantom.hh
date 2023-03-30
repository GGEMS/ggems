#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSMESHEDPHANTOM_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSMESHEDPHANTOM_HH

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
  \file GGEMSMeshedPhantom.hh

  \brief Child GGEMS class handling meshed phantom

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday June 14, 2022
*/

#include "GGEMS/navigators/GGEMSNavigator.hh"

/*!
  \class GGEMSMeshedPhantom
  \brief Child GGEMS class handling meshed phantom
*/
class GGEMS_EXPORT GGEMSMeshedPhantom : public GGEMSNavigator
{
  public:
    /*!
      \param meshed_phantom_name - name of the meshed phantom
      \brief GGEMSMeshedPhantom constructor
    */
    explicit GGEMSMeshedPhantom(std::string const& meshed_phantom_name);

    /*!
      \brief GGEMSMeshedPhantom destructor
    */
    ~GGEMSMeshedPhantom(void) override;

    /*!
      \fn GGEMSMeshedPhantom(GGEMSMeshedPhantom const& meshed_phantom_name) = delete
      \param meshed_phantom_name - reference on the GGEMS meshed phantom
      \brief Avoid copy by reference
    */
    GGEMSMeshedPhantom(GGEMSMeshedPhantom const& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const& meshed_phantom_name) = delete
      \param meshed_phantom_name - reference on the GGEMS meshed phantom
      \brief Avoid assignement by reference
    */
    GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete
      \param meshed_phantom_name - rvalue reference on the GGEMS meshed phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedPhantom(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete;

    /*!
      \fn GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete
      \param meshed_phantom_name - rvalue reference on the GGEMS meshed phantom
      \brief Avoid copy by rvalue reference
    */
    GGEMSMeshedPhantom& operator=(GGEMSMeshedPhantom const&& meshed_phantom_name) = delete;

    /*!
      \fn void SetPhantomFile(std::string const& voxelized_phantom_filename, std::string const& range_data_filename)
      \param voxelized_phantom_filename - MHD filename for voxelized phantom
      \param range_data_filename - text file with range to material data
      \brief set the mhd filename for voxelized phantom and the range data file
    */
    void SetPhantomFile(std::string const& meshed_phantom_filename);

    /*!
      \fn void Initialize(void) override
      \brief Initialize the meshed phantom
    */
    void Initialize(void) override;

    /*!
      \fn void SaveResults
      \brief save all results from solid
    */
    void SaveResults(void) override;

  private:
    /*!
      \fn void CheckParameters(void) const override
      \brief checking parameters
    */
    void CheckParameters(void) const override;

  private:
    std::string meshed_phantom_filename_; /*!< Mesh file storing the meshed phantom */
};

/*!
  \fn GGEMSMeshedPhantom* create_ggems_meshed_phantom(char const* meshed_phantom_name)
  \param meshed_phantom_name - name of meshed phantom
  \return the pointer on the meshed phantom
  \brief Get the GGEMSMeshedPhantom pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMeshedPhantom* create_ggems_meshed_phantom(char const* meshed_phantom_name);

/*!
  \fn void set_phantom_file_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* phantom_filename)
  \param meshed_phantom - pointer on meshed_phantom
  \param phantom_filename - filename of the meshed phantom
  \brief set the filename of meshed phantom
*/
extern "C" GGEMS_EXPORT void set_phantom_file_ggems_meshed_phantom(GGEMSMeshedPhantom* meshed_phantom, char const* phantom_filename);

#endif
