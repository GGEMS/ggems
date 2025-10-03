#ifndef GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH

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
  \file GGEMSCrossSections.hh

  \brief GGEMS class handling the cross sections tables

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

/// \cond
#include <vector>
#include <string>
/// \endcond

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/physics/GGEMSParticleCrossSections.hh"

class GGEMSEMProcess;
class GGEMSMaterials;
class GGEMSProcessesManager;

/*!
  \class GGEMSCrossSections
  \brief GGEMS class handling the cross sections tables
*/
class GGEMS_EXPORT GGEMSCrossSections
{
  public:
    /*!
      \param materials - pointer to materials
      \brief GGEMSCrossSections constructor
    */
    explicit GGEMSCrossSections(GGEMSMaterials* materials);

    /*!
      \brief GGEMSCrossSections destructor
    */
    ~GGEMSCrossSections(void);

    /*!
      \fn GGEMSCrossSections(GGEMSCrossSections const& cross_sections) = delete
      \param cross_sections - reference on the GGEMS cross sections
      \brief Avoid copy by reference
    */
    GGEMSCrossSections(GGEMSCrossSections const& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections& operator=(GGEMSCrossSections const& cross_sections) = delete
      \param cross_sections - reference on the GGEMS cross sections
      \brief Avoid assignement by reference
    */
    GGEMSCrossSections& operator=(GGEMSCrossSections const& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections(GGEMSCrossSections const&& cross_sections) = delete
      \param cross_sections - rvalue reference on the GGEMS cross sections
      \brief Avoid copy by rvalue reference
    */
    GGEMSCrossSections(GGEMSCrossSections const&& cross_sections) = delete;

    /*!
      \fn GGEMSCrossSections& operator=(GGEMSCrossSections const&& cross_sections) = delete
      \param cross_sections - rvalue reference on the GGEMS cross sections
      \brief Avoid copy by rvalue reference
    */
    GGEMSCrossSections& operator=(GGEMSCrossSections const&& cross_sections) = delete;

    /*!
      \fn void AddProcess(std::string const& process_name, std::string const& particle_type, bool const& is_secondary)
      \param process_name - name of the process
      \param particle_type - type of the particle
      \param is_secondary - activate secondaries or not
      \brief add a process to the GGEMS simulation
    */
    void AddProcess(std::string const& process_name, std::string const& particle_type, bool const& is_secondary = false);

    /*!
      \fn void Initialize(void)
      \brief Initialize all the activated processes computing tables on OpenCL device
    */
    void Initialize(void);

    /*!
      \fn inline GGEMSEMProcess** GetEMProcessesList(void) const
      \return pointer to process list
      \brief get the pointer on activated process
    */
    inline GGEMSEMProcess** GetEMProcessesList(void) const {return em_processes_list_;}

    /*!
      \fn inline GGsize GetNumberOfActivatedEMProcesses(void) const
      \return number of activated processes
      \brief get the number of activated processes
    */
    inline GGsize GetNumberOfActivatedEMProcesses(void) const {return number_of_activated_processes_;}

    /*!
      \fn inline cl::Buffer* GetCrossSections(GGsize const& thread_index) const
      \param thread_index - index of activated device (thread index)
      \return pointer to OpenCL buffer storing cross sections
      \brief return the pointer to OpenCL buffer storing cross sections
    */
    inline cl::Buffer* GetCrossSections(GGsize const& thread_index) const {return particle_cross_sections_[thread_index];}

    /*!
      \fn GGfloat GetPhotonCrossSection(std::string const& process_name, std::string const& material_name, GGfloat const& energy, std::string const& unit) const
      \param process_name - name of the process
      \param material_name - name of the material
      \param energy - energy of particle
      \param unit - unit in energy
      \return the cross section in cm2.g-1 for a process and a material
      \brief Get the cross section value for a process for a specific energy
    */
    GGfloat GetPhotonCrossSection(std::string const& process_name, std::string const& material_name, GGfloat const& energy, std::string const& unit) const;

    /*!
      \fn void Clean(void)
      \brief clean all cross sections on each OpenCL device
    */
    void Clean(void);

  private:
    /*!
      \fn void LoadPhysicTablesOnHost(void)
      \brief Load physic tables from OpenCL device to RAM. Optimization for python user
    */
    void LoadPhysicTablesOnHost(void);

  private:
    GGEMSEMProcess** em_processes_list_; /*!< vector of electromagnetic processes */
    GGsize number_of_activated_processes_; /*!< Number of activated processes */
    std::vector<bool> is_process_activated_; /*!< Boolean checking if the process is already activated */
    cl::Buffer** particle_cross_sections_; /*!< Pointer storing cross sections for each particles on OpenCL device */
    GGEMSParticleCrossSections* particle_cross_sections_host_; /*!< Pointer storing cross sections for each particles on host (RAM memory) */
    GGsize number_activated_devices_; /*!< Number of activated device */
    GGEMSMaterials* materials_; /*!< Pointer to material defined in a navigator */
};

/*!
  \fn GGEMSCrossSections* create_ggems_cross_sections(GGEMSMaterials* materials)
  \param materials - pointer to materials
  \return the pointer on the singleton
  \brief Get the GGEMSCrossSections pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSCrossSections* create_ggems_cross_sections(GGEMSMaterials* materials);

/*!
  \fn void add_process_ggems_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* particle_name, bool const is_secondary)
  \param cross_sections - pointer on GGEMS cross sections
  \param process_name - name of the process
  \param particle_name - name of the particle
  \param is_secondary - activate secondaries or not
  \brief Add a process to cross section table
*/
extern "C" GGEMS_EXPORT void add_process_ggems_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* particle_name, bool const is_secondary);

/*!
  \fn void initialize_ggems_cross_sections(GGEMSCrossSections* cross_sections)
  \param cross_sections - pointer on GGEMS cross sections
  \brief Intialize the cross section tables for process and materials
*/
extern "C" GGEMS_EXPORT void initialize_ggems_cross_sections(GGEMSCrossSections* cross_sections);

/*!
  \fn GGfloat get_cs_ggems_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* material_name, GGfloat const energy, char const* unit)
  \param cross_sections - pointer on GGEMS cross sections
  \param process_name - name of the process
  \param material_name - name of the material
  \param energy - energy of the particle
  \param unit - unit in energy
  \return cross section value in cm2.g-1
  \brief get the cross section value of process
*/
extern "C" GGEMS_EXPORT GGfloat get_cs_ggems_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* material_name, GGfloat const energy, char const* unit);

/*!
  \fn void clean_ggems_cross_sections(GGEMSCrossSections* cross_sections)
  \param cross_sections - pointer on GGEMS cross sections
  \brief clean all cross sections on each OpenCL device
*/
extern "C" GGEMS_EXPORT void clean_ggems_cross_sections(GGEMSCrossSections* cross_sections);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
