#ifndef GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH

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

#include <vector>
#include <string>

#include "GGEMS/global/GGEMSOpenCLManager.hh"

class GGEMSEMProcess;
class GGEMSMaterials;
class GGEMSProcessesManager;

typedef std::vector<std::shared_ptr<GGEMSEMProcess>> GGEMSEMProcessesList; /*!< vector of pointer storing physical processes */

/*!
  \class GGEMSCrossSections
  \brief GGEMS class handling the cross sections tables
*/
class GGEMS_EXPORT GGEMSCrossSections
{
  public:
    /*!
      \brief GGEMSCrossSections constructor
    */
    GGEMSCrossSections(void);

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
      \fn void Initialize(GGEMSMaterials const* materials)
      \param materials - activated materials for a specific phantom
      \brief Initialize all the activated processes computing tables on OpenCL device
    */
    void Initialize(GGEMSMaterials const* materials);

    /*!
      \fn inline GGEMSEMProcessesList GetProcessesList(void) const
      \return pointer to process list
      \brief get the pointer on activated process
    */
    inline GGEMSEMProcessesList GetProcessesList(void) const {return em_processes_list_;}

    /*!
      \fn GGfloat GetPhotonCrossSection(std::string const& process_name, std::string const& material_name, GGfloat const& energy, std::string const& unit) const
      \param process_name - name of the process
      \param material_name - name of the material
      \param energy - energy of particle
      \param unit - unit in energy
      \return the cross section in cm-1 for a process and a material
      \brief Get the cross section value for a process for a specific energy
    */
    GGfloat GetPhotonCrossSection(std::string const& process_name, std::string const& material_name, GGfloat const& energy, std::string const& unit) const;

  private:
    GGEMSEMProcessesList em_processes_list_; /*!< vector of electromagnetic processes */
    std::vector<bool> is_process_activated_; /*!< Boolean checking if the process is already activated */
    std::shared_ptr<cl::Buffer> particle_cross_sections_; /*!< Pointer storing cross sections for each particles */
};

/*!
  \fn GGEMSCrossSections* create_ggems_cross_sections(void)
  \return the pointer on the singleton
  \brief Get the GGEMSCrossSections pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSCrossSections* create_ggems_cross_sections(void);

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
  \fn void initialize_ggems_cross_sections(GGEMSCrossSections* cross_sections, GGEMSMaterials* materials)
  \param cross_sections - pointer on GGEMS cross sections
  \param materials - pointer on GGEMS materials
  \brief Intialize the cross section tables for process and materials
*/
extern "C" GGEMS_EXPORT void initialize_ggems_cross_sections(GGEMSCrossSections* cross_sections, GGEMSMaterials* materials);

/*!
  \fn GGfloat get_cs_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* material_name, GGfloat const energy, char const* unit)
  \param cross_sections - pointer on GGEMS cross sections
  \param process_name - name of the process
  \param material_name - name of the material
  \param energy - energy of the particle
  \param unit - unit in energy
  \return cross section value in cm-1
  \brief get the cross section value of process
*/
extern "C" GGEMS_EXPORT GGfloat get_cs_cross_sections(GGEMSCrossSections* cross_sections, char const* process_name, char const* material_name, GGfloat const energy, char const* unit);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSCROSSSECTIONS_HH
