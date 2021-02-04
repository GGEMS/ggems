#ifndef GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH

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
  \file GGEMSEMProcess.hh

  \brief GGEMS mother class for electromagnectic process

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 30, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/materials/GGEMSMaterialTables.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/physics/GGEMSParticleCrossSections.hh"

/*!
  \class GGEMSEMProcess
  \brief GGEMS mother class for electromagnectic process
*/
class GGEMS_EXPORT GGEMSEMProcess
{
  public:
    /*!
      \brief GGEMSEMProcess constructor
    */
    GGEMSEMProcess(void);

    /*!
      \brief GGEMSEMProcess destructor
    */
    virtual ~GGEMSEMProcess(void);

    /*!
      \fn GGEMSEMProcess(GGEMSEMProcess const& em_process) = delete
      \param em_process - reference on the GGEMS electromagnetic process
      \brief Avoid copy by reference
    */
    GGEMSEMProcess(GGEMSEMProcess const& em_process) = delete;

    /*!
      \fn GGEMSEMProcess& operator=(GGEMSEMProcess const& em_process) = delete
      \param em_process - reference on the GGEMS electromagnetic process
      \brief Avoid assignement by reference
    */
    GGEMSEMProcess& operator=(GGEMSEMProcess const& em_process) = delete;

    /*!
      \fn GGEMSEMProcess(GGEMSEMProcess const&& em_process) = delete
      \param em_process - rvalue reference on the GGEMS electromagnetic process
      \brief Avoid copy by rvalue reference
    */
    GGEMSEMProcess(GGEMSEMProcess const&& em_process) = delete;

    /*!
      \fn GGEMSEMProcess& operator=(GGEMSEMProcess const&& em_process) = delete
      \param em_process - rvalue reference on the GGEMS electromagnetic process
      \brief Avoid copy by rvalue reference
    */
    GGEMSEMProcess& operator=(GGEMSEMProcess const&& em_process) = delete;

    /*!
      \fn inline std::string GetProcessName(void) const
      \return name of the process
      \brief get the name of the process
    */
    inline std::string GetProcessName(void) const {return process_name_;}

    /*!
      \fn void BuildCrossSectionTables(std::weak_ptr<cl::Buffer> particle_cross_sections, std::weak_ptr<cl::Buffer> material_tables)
      \param particle_cross_sections - OpenCL buffer storing all the cross section tables for each particles
      \param material_tables - material tables on OpenCL device
      \brief build cross section tables and storing them in particle_cross_sections
    */
    virtual void BuildCrossSectionTables(std::weak_ptr<cl::Buffer> particle_cross_sections, std::weak_ptr<cl::Buffer> material_tables);

  protected:
    /*!
      \fn GGfloat ComputeCrossSectionPerMaterial(GGEMSParticleCrossSections* cross_section, GGEMSMaterialTables const* material_tables, GGsize const& material_index, GGsize const& energy_index)
      \param cross_section - cross section
      \param material_tables - activated material for a phantom
      \param material_index - index of the material
      \param energy_index - index of the energy
      \return cross section for a process for a material
      \brief compute cross section for a process for a material
    */
    GGfloat ComputeCrossSectionPerMaterial(GGEMSParticleCrossSections* cross_section, GGEMSMaterialTables const* material_tables, GGsize const& material_index, GGsize const& energy_index);

    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return cross section by atom
      \brief compute a cross section for an atom
    */
    virtual GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const = 0;

  protected:
    GGchar process_id_; /*!< Id of the process as defined in GGEMSEMProcessConstants.hh */
    std::string process_name_; /*!< Name of the process */
    std::string primary_particle_; /*!< Type of primary particle */
    std::string secondary_particle_; /*!< Type of secondary particle */
    bool is_secondaries_; /*!< Flag to activate secondaries */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
