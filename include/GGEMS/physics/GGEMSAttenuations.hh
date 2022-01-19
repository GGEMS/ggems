#ifndef GUARD_GGEMS_PHYSICS_GGEMSATTENUATIONS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSATTENUATIONS_HH

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
  \file GGEMSAttenuations.hh

  \brief Class computing and storing attenuation coefficient

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author Mateo VILLA <ingmatvillaa@gmail.com>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \date Tuesday January 18, 2022
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/physics/GGEMSMuData.hh"

class GGEMSMaterials;
class GGEMSCrossSections;

/*!
  \class GGEMSAttenuations
  \brief Class computing and storing attenuation coefficient
*/
class GGEMS_EXPORT GGEMSAttenuations
{
  public:
    /*!
      \param materials - activated materials for a specific phantom
      \brief GGEMSAttenuations constructor
    */
    explicit GGEMSAttenuations(GGEMSMaterials* materials);

    /*!
      \brief GGEMSAttenuations destructor
    */
    ~GGEMSAttenuations(void);

    /*!
      \fn GGEMSAttenuations(GGEMSAttenuations const& attenuations) = delete
      \param attenuations - reference on the GGEMS attenuations
      \brief Avoid copy by reference
    */
    GGEMSAttenuations(GGEMSAttenuations const& attenuations) = delete;

    /*!
      \fn GGEMSAttenuations& operator=(GGEMSAttenuations const& attenuations) = delete
      \param attenuations - reference on the GGEMS attenuations
      \brief Avoid assignement by reference
    */
    GGEMSAttenuations& operator=(GGEMSAttenuations const& attenuations) = delete;

    /*!
      \fn GGEMSAttenuations(GGEMSAttenuations const&& attenuations) = delete
      \param attenuations - rvalue reference on the GGEMS attenuations
      \brief Avoid copy by rvalue reference
    */
    GGEMSAttenuations(GGEMSAttenuations const&& attenuations) = delete;

    /*!
      \fn GGEMSAttenuations& operator=(GGEMSAttenuations const&& attenuations) = delete
      \param attenuations - rvalue reference on the GGEMS attenuations
      \brief Avoid copy by rvalue reference
    */
    GGEMSAttenuations& operator=(GGEMSAttenuations const&& attenuations) = delete;

    /*!
      \fn void Initialize(GGEMSCrossSections const* cross_sections)
      \param cross_sections - stored cross sections for a specific navigator
      \brief Initialize all the activated processes computing tables on OpenCL device
    */
    void Initialize(GGEMSCrossSections const* cross_sections);

    /*!
      \fn void Clean(void)
      \brief clean all OpenCL buffer
    */
    void Clean(void);

    /*!
      \fn inline cl::Buffer* GetAttenuations(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to OpenCL buffer storing attenuations
      \brief return the pointer to OpenCL buffer storing attenuations
    */
    inline cl::Buffer* GetAttenuations(GGsize const& thread_index) const {return mu_tables_[thread_index];}

    /*!
      \fn GGfloat GetAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const
      \param material_name - name of the material
      \param energy - energy of particle
      \param unit - unit in energy
      \return attenuation in cm-1 for a material
      \brief Get the attenuation coefficient value for a material for a specific energy
    */
    GGfloat GetAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const;

    /*!
      \fn GGfloat GetEnergyAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const
      \param material_name - name of the material
      \param energy - energy of particle
      \param unit - unit in energy
      \return energy attenuation in cm-1 for a material
      \brief Get the energy attenuation coefficient value for a material for a specific energy
    */
    GGfloat GetEnergyAttenuation(std::string const& material_name, GGfloat const& energy, std::string const& unit) const;

  private:
    /*!
      \fn void LoadAttenuationsOnHost(void)
      \brief Load attenuations coefficients from OpenCL device to RAM. Optimization for python user
    */
    void LoadAttenuationsOnHost(void);

  private:
    GGsize number_activated_devices_; /*!< Number of activated device */

    GGfloat* energies_; /*!< energy values */
    GGfloat* mu_; /*!< attenuation coefficients */
    GGfloat* mu_en_; /*!< energy-absorption coefficient */
    GGint* mu_index_; /*!< index of attenuation */
    GGEMSMuMuEnData* attenuations_host_; /*!< Pointer storing attenuations coef. on host (RAM memory) */
    GGEMSMaterials* materials_; /*!< pointer to materials */

    // OpenCL Buffer
    cl::Buffer** mu_tables_; /*<! attenuations coefficients on OpenCL device */
};

#endif
