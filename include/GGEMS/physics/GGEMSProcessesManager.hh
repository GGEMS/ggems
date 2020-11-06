#ifndef GUARD_GGEMS_PHYSICS_GGEMSPROCESSESMANAGER_HH
#define GUARD_GGEMS_PHYSICS_GGEMSPROCESSESMANAGER_HH

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
  \file GGEMSProcessesManager.hh

  \brief GGEMS class managing the processes in GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday March 9, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/physics/GGEMSProcessConstants.hh"

/*!
  \class GGEMSProcessesManager
  \brief GGEMS class managing the processes in GGEMS simulation
*/
class GGEMS_EXPORT GGEMSProcessesManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSProcessesManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSProcessesManager(void);

  public:
    /*!
      \fn static GGEMSProcessesManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSProcessesManager
    */
    static GGEMSProcessesManager& GetInstance(void)
    {
      static GGEMSProcessesManager instance;
      return instance;
    }

    /*!
      \fn GGEMSProcessesManager(GGEMSProcessesManager const& processes_manager) = delete
      \param processes_manager - reference on the processes manager
      \brief Avoid copy of the class by reference
    */
    GGEMSProcessesManager(GGEMSProcessesManager const& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager& operator=(GGEMSProcessesManager const& processes_manager) = delete
      \param processes_manager - reference on the processes manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSProcessesManager& operator=(GGEMSProcessesManager const& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager(GGEMSProcessesManager const&& processes_manager) = delete
      \param processes_manager - rvalue reference on the processes manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProcessesManager(GGEMSProcessesManager const&& processes_manager) = delete;

    /*!
      \fn GGEMSProcessesManager& operator=(GGEMSProcessesManager const&& processes_manager) = delete
      \param processes_manager - rvalue reference on the processes manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSProcessesManager& operator=(GGEMSProcessesManager const&& processes_manager) = delete;

    /*!
      \fn void SetCrossSectionTableNumberOfBins(GGshort const& number_of_bins)
      \param number_of_bins - number of bins in cross section table
      \brief set the number of bins in the cross section table
    */
    void SetCrossSectionTableNumberOfBins(GGshort const& number_of_bins);

    /*!
      \fn void SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit = "keV")
      \param energy - minimum energy in the cross section table
      \param unit - unit of energy
      \brief set the minimum energy in the cross section table
    */
    void SetCrossSectionTableMinimumEnergy(GGfloat const& energy, char const* unit = "keV");

    /*!
      \fn void SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit = "keV")
      \param energy - maximum energy in the cross section table
      \param unit - unit of energy
      \brief set the maximum energy in the cross section table
    */
    void SetCrossSectionTableMaximumEnergy(GGfloat const& energy, char const* unit = "keV");

    /*!
      \fn inline GGfloat GetCrossSectionTableMinEnergy(void) const
      \return the minimum energy in the cross section table
      \brief get the minimum energy in the cross section table
    */
    inline GGfloat GetCrossSectionTableMinEnergy(void) const {return cross_section_table_min_energy_;}

    /*!
      \fn inline GGfloat GetCrossSectionTableMaxEnergy(void) const
      \return the maximum energy in the cross section table
      \brief get the maximum energy in the cross section table
    */
    inline GGfloat GetCrossSectionTableMaxEnergy(void) const {return cross_section_table_max_energy_;}

    /*!
      \fn inline GGshort GetCrossSectionTableNumberOfBins(void) const
      \return the number of bins in the cross section table
      \brief get the number of bins in the cross section table
    */
    inline GGshort GetCrossSectionTableNumberOfBins(void) const {return cross_section_table_number_of_bins_;}

    /*!
      \fn void AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name)
      \param process_name - Name of the process
      \param particle_name - Name of the particle
      \param phantom_name - Name of the phantom
      \brief add a process for a specific phantom or all the phantom
    */
    void AddProcess(std::string const& process_name, std::string const& particle_name, std::string const& phantom_name);

    /*!
      \fn void PrintInfos(void) const
      \brief Print all infos about processes
    */
    void PrintInfos(void) const;

    /*!
      \fn void PrintAvailableProcesses(void) const
      \brief Print all infos about available processes
    */
    void PrintAvailableProcesses(void) const;

    /*!
      \fn void PrintPhysicTables(bool const& is_processes_print_tables)
      \param is_processes_print_tables - Flag for physic tables printing
      \brief print physic tables to screen
    */
    void PrintPhysicTables(bool const& is_processes_print_tables);

    /*!
      \fn bool IsPrintPhysicTables(void) const
      \brief check boolean value for physic tables printing
    */
    inline bool IsPrintPhysicTables(void) const {return is_processes_print_tables_;};

  private:
    GGshort cross_section_table_number_of_bins_; /*!< Number of bins in the cross section table */
    GGfloat cross_section_table_min_energy_; /*!< Minimum energy in the cross section table */
    GGfloat cross_section_table_max_energy_; /*!< Maximum energy in the cross section table */
    bool is_processes_print_tables_; /*!< Flag for physic tables printing */
};

/*!
  \fn GGEMSProcessesManager* get_instance_processes_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSProcessesManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSProcessesManager* get_instance_processes_manager(void);

/*!
  \fn void add_process_processes_manager(GGEMSProcessesManager* processes_manager, char const* process_name, char const* particle_name, char const* phantom_name)
  \param processes_manager - pointer on the processes manager
  \param process_name - Name of the process
  \param particle_name - Name of the particle
  \param phantom_name - Name of the phantom
  \brief add a process for a specific phantom or all the phantom
*/
extern "C" GGEMS_EXPORT void add_process_processes_manager(GGEMSProcessesManager* processes_manager, char const* process_name, char const* particle_name, char const* phantom_name);

/*!
  \fn void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGshort const number_of_bins)
  \param processes_manager - pointer on the processes manager
  \param number_of_bins - number of the bins for the cross section table
  \brief set the number of the bins in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_number_of_bins_processes_manager(GGEMSProcessesManager* processes_manager, GGshort const number_of_bins);

/*!
  \fn void set_cross_section_table_minimum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
  \param processes_manager - pointer on the processes manager
  \param energy - minimum energy
  \param unit - unit of energy
  \brief set the minimum energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_minimum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit);

/*!
  \fn void set_cross_section_table_maximum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit)
  \param processes_manager - pointer on the processes manager
  \param energy - maximum energy
  \param unit - unit of energy
  \brief set the maximum energy in the cross section table
*/
extern "C" GGEMS_EXPORT void set_cross_section_table_maximum_energy_processes_manager(GGEMSProcessesManager* processes_manager, GGfloat const energy, char const* unit);

/*!
  \fn void print_infos_processes_manager(GGEMSProcessesManager* processes_manager)
  \param processes_manager - pointer on the processes manager
  \brief print infos about processes
*/
extern "C" GGEMS_EXPORT void print_infos_processes_manager(GGEMSProcessesManager* processes_manager);

/*!
  \fn void print_available_processes_manager(GGEMSProcessesManager* processes_manager)
  \param processes_manager - pointer on the processes manager
  \brief print infos about available processes
*/
extern "C" GGEMS_EXPORT void print_available_processes_manager(GGEMSProcessesManager* processes_manager);

/*!
  \fn void print_available_processes_manager(GGEMSProcessesManager* processes_manager, bool const is_processes_print_tables)
  \param processes_manager - pointer on the processes manager
  \param is_processes_print_tables - flag printing physic tables
  \brief print infos about physic tables
*/
extern "C" GGEMS_EXPORT void print_tables_processes_manager(GGEMSProcessesManager* processes_manager, bool const is_processes_print_tables);

#endif // GUARD_GGEMS_PHYSICS_GGEMSRANGECUTSMANAGER_HH
