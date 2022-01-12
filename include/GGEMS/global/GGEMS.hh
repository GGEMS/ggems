#ifndef GUARD_GGEMS_GLOBAL_GGEMS_HH
#define GUARD_GGEMS_GLOBAL_GGEMS_HH

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
  \file GGEMS.hh

  \brief GGEMS class managing the GGEMS simulation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday September 30, 2019
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <cstdint>
#include <string>
#include <vector>

#include "GGEMS/global/GGEMSExport.hh"

#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMS
  \brief GGEMS class managing the complete simulation
*/
class GGEMS_EXPORT GGEMS
{
  public:
    /*!
      \brief GGEMS constructor
    */
    GGEMS(void);

    /*!
      \brief GGEMS destructor
    */
    ~GGEMS(void);

    /*!
      \fn GGEMS(GGEMS const& ggems) = delete
      \param ggems - reference on the ggems
      \brief Avoid copy of the class by reference
    */
    GGEMS(GGEMS const& ggems) = delete;

    /*!
      \fn GGEMS& operator=(GGEMS const& ggems) = delete
      \param ggems - reference on the ggems
      \brief Avoid assignement of the class by reference
    */
    GGEMS& operator=(GGEMS const& ggems) = delete;

    /*!
      \fn GGEMS(GGEMS const&& ggems) = delete
      \param ggems - rvalue reference on the ggems
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMS(GGEMS const&& ggems) = delete;

    /*!
      \fn GGEMS& operator=(GGEMS const&& ggems) = delete
      \param ggems - rvalue reference on the ggems
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMS& operator=(GGEMS const&& ggems) = delete;

    /*!
      \fn void Initialize(GGuint const& seed = 0)
      \param seed - seed of the simulation
      \brief Initialization of the GGEMS simulation
    */
    void Initialize(GGuint const& seed = 0);

    /*!
      \fn void Run(void)
      \brief run the GGEMS simulation
    */
    void Run(void);

    /*!
      \fn void SetOpenCLVerbose(bool const& is_opencl_verbose)
      \param is_opencl_verbose - flag for opencl verbosity
      \brief set the flag for OpenCL verbosity
    */
    void SetOpenCLVerbose(bool const& is_opencl_verbose);

    /*!
      \fn void SetMaterialDatabaseVerbose(bool const& is_material_database_verbose)
      \param is_material_database_verbose - flag for material database verbosity
      \brief set the flag for material database verbosity
    */
    void SetMaterialDatabaseVerbose(bool const& is_material_database_verbose);

    /*!
      \fn void SetSourceVerbose(bool const& is_source_verbose)
      \param is_source_verbose - flag for source verbosity
      \brief set the flag for source verbosity
    */
    void SetSourceVerbose(bool const& is_source_verbose);

    /*!
      \fn void SetNavigatorVerbose(bool const is_navigator_verbose)
      \param is_navigator_verbose - flag for navigator verbosity
      \brief set the flag for navigator verbosity
    */
    void SetNavigatorVerbose(bool const& is_navigator_verbose);

    /*!
      \fn void SetMemoryRAMVerbose(bool const& is_memory_ram_verbose)
      \param is_memory_ram_verbose - flag for memory RAM verbosity
      \brief set the flag for memory RAM verbosity
    */
    void SetMemoryRAMVerbose(bool const& is_memory_ram_verbose);

    /*!
      \fn void SetProcessVerbose(bool const& is_process_verbose)
      \param is_process_verbose - flag for process verbosity
      \brief set the flag for process verbosity
    */
    void SetProcessVerbose(bool const& is_process_verbose);

    /*!
      \fn void SetRangeCutsVerbose(bool const& is_range_cuts_verbose)
      \param is_range_cuts_verbose - flag for range cuts verbosity
      \brief set the flag for range cuts verbosity
    */
    void SetRangeCutsVerbose(bool const& is_range_cuts_verbose);

    /*!
      \fn void SetRandomVerbose(bool const& is_random_verbose)
      \param is_random_verbose - flag for random verbosity
      \brief set the flag for random verbosity
    */
    void SetRandomVerbose(bool const& is_random_verbose);

    /*!
      \fn void SetProfilingVerbose(bool const& is_profiling_verbose)
      \param is_profiling_verbose - flag for profiling timer verbosity
      \brief set the flag for profiling timer verbosity
    */
    void SetProfilingVerbose(bool const& is_profiling_verbose);

    /*!
      \fn bool IsProfilingVerbose(void) const
      \return state of profiling verbosity flag
      \brief get the profiling verbosity flag
    */
    inline bool IsProfilingVerbose(void) const {return is_profiling_verbose_;}

    /*!
      \fn void SetTrackingVerbose(bool const& is_tracking_verbose, GGint const& particle_tracking_id)
      \param is_tracking_verbose - flag for tracking verbosity
      \param particle_tracking_id - particle id for tracking
      \brief set the flag for tracking verbosity and an index for particle tracking
    */
    void SetTrackingVerbose(bool const& is_tracking_verbose, GGint const& particle_tracking_id);

    /*!
      \fn bool IsTrackingVerbose(void) const
      \return state of tracking verbosity flag
      \brief get the tracking verbosity flag
    */
    inline bool IsTrackingVerbose(void) const {return is_tracking_verbose_;}

    /*!
      \fn GGint GetParticleTrackingID(void) const
      \return id of the particle to track
      \brief get the id of the particle to track
    */
    inline GGint GetParticleTrackingID(void) const {return particle_tracking_id_;}

  private:
    /*!
      \fn void PrintBanner(void) const
      \brief Print GGEMS banner
    */
    void PrintBanner(void) const;

    /*!
      \fn void RunOnDevice(GGsize const& thread_index)
      \param thread_index - index of the thread
      \brief run the GGEMS simulation on each thread associated to a OpenCL device
    */
    void RunOnDevice(GGsize const& thread_index);

  private: // Global simulation parameters
    bool is_opencl_verbose_; /*!< Flag for OpenCL verbosity */
    bool is_material_database_verbose_; /*!< Flag for material database verbosity */
    bool is_source_verbose_; /*!< Flag for source verbosity */
    bool is_navigator_verbose_; /*!< Flag for navigator verbosity */
    bool is_memory_ram_verbose_; /*!< Flag for memory RAM verbosity */
    bool is_process_verbose_; /*!< Flag for processes verbosity */
    bool is_range_cuts_verbose_; /*!< Flag for range cuts verbosity */
    bool is_random_verbose_; /*!< Flag for random verbosity */
    bool is_tracking_verbose_; /*!< Flag for tracking verbosity */
    bool is_profiling_verbose_; /*!< Flag for kernel time verbosity */
    GGint particle_tracking_id_; /*!< Particle if for tracking */
};

/*!
  \fn GGEMS* create_ggems(void)
  \return the pointer to GGEMS
  \brief Get the GGEMS pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMS* create_ggems(void);

/*!
  \fn GGEMSBox* delete_ggems(GGEMS* ggems)
  \param ggems - pointer on ggems
  \brief Delete GGEMS pointer
*/
extern "C" GGEMS_EXPORT void delete_ggems(GGEMS* ggems);

/*!
  \fn void initialize_ggems(GGEMS* ggems, GGuint const seed)
  \param ggems - pointer to GGEMS
  \param seed - seed of the random
  \brief Initialize GGEMS simulation
*/
extern "C" GGEMS_EXPORT void initialize_ggems(GGEMS* ggems, GGuint const seed);

/*!
  \fn void set_opencl_verbose_ggems(GGEMS* ggems, bool const is_opencl_verbose)
  \param ggems - pointer to GGEMS
  \param is_opencl_verbose - flag on opencl verbose
  \brief Set the OpenCL verbosity
*/
extern "C" GGEMS_EXPORT void set_opencl_verbose_ggems(GGEMS* ggems, bool const is_opencl_verbose);

/*!
  \fn void set_material_database_verbose_ggems(GGEMS* ggems, bool const is_material_database_verbose)
  \param ggems - pointer to GGEMS
  \param is_material_database_verbose - flag on material database verbose
  \brief Set the material database verbosity
*/
extern "C" GGEMS_EXPORT void set_material_database_verbose_ggems(GGEMS* ggems, bool const is_material_database_verbose);

/*!
  \fn void set_source_ggems(GGEMS* ggems, bool const is_source_verbose)
  \param ggems - pointer to GGEMS
  \param is_source_verbose - flag on source verbose
  \brief Set the source verbosity
*/
extern "C" GGEMS_EXPORT void set_source_ggems(GGEMS* ggems, bool const is_source_verbose);

/*!
  \fn void set_navigator_ggems(GGEMS* ggems, bool const is_navigator_verbose)
  \param ggems - pointer to GGEMS
  \param is_navigator_verbose - flag on navigator verbose
  \brief Set the navigator verbosity
*/
extern "C" GGEMS_EXPORT void set_navigator_ggems(GGEMS* ggems, bool const is_navigator_verbose);

/*!
  \fn void set_memory_ram_ggems(GGEMS* ggems, bool const is_memory_ram_verbose)
  \param ggems - pointer to GGEMS
  \param is_memory_ram_verbose - flag on memory RAM verbose
  \brief Set the memory RAM verbosity
*/
extern "C" GGEMS_EXPORT void set_memory_ram_ggems(GGEMS* ggems, bool const is_memory_ram_verbose);

/*!
  \fn void set_process_ggems(GGEMS* ggems, bool const is_process_verbose)
  \param ggems - pointer to GGEMS
  \param is_process_verbose - flag on processes verbose
  \brief Set the processes verbosity
*/
extern "C" GGEMS_EXPORT void set_process_ggems(GGEMS* ggems, bool const is_process_verbose);

/*!
  \fn void set_range_cuts_ggems(GGEMS* ggems, bool const is_range_cuts_verbose)
  \param ggems - pointer to GGEMS
  \param is_range_cuts_verbose - flag on range cuts verbose
  \brief Set the range cuts verbosity
*/
extern "C" GGEMS_EXPORT void set_range_cuts_ggems(GGEMS* ggems, bool const is_range_cuts_verbose);

/*!
  \fn void set_random_ggems(GGEMS* ggems, bool const is_random_verbose)
  \param ggems - pointer to GGEMS
  \param is_random_verbose - flag on random verbose
  \brief Set the random verbosity
*/
extern "C" GGEMS_EXPORT void set_random_ggems(GGEMS* ggems, bool const is_random_verbose);

/*!
  \fn void set_profiling_ggems(GGEMS* ggems, bool const is_profiling_verbose)
  \param ggems - pointer to GGEMS
  \param is_profiling_verbose - flag on profiling verbose
  \brief Set the profiling verbosity
*/
extern "C" GGEMS_EXPORT void set_profiling_ggems(GGEMS* ggems, bool const is_profiling_verbose);

/*!
  \fn void set_tracking_ggems(GGEMS* ggems, bool const is_tracking_verbose, GGint const particle_id_tracking)
  \param ggems - pointer to GGEMS
  \param is_tracking_verbose - flag on tracking verbose
  \param particle_id_tracking - particle id for tracking
  \brief Set the tracking verbosity
*/
extern "C" GGEMS_EXPORT void set_tracking_ggems(GGEMS* ggems, bool const is_tracking_verbose, GGint const particle_id_tracking);

/*!
  \fn void run_ggems(GGEMS* ggems)
  \param ggems - pointer to GGEMS
  \brief Run the GGEMS simulation
*/
extern "C" GGEMS_EXPORT void run_ggems(GGEMS* ggems);

#endif // End of GUARD_GGEMS_GLOBAL_GGEMS_HH
