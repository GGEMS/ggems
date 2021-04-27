#ifndef GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH
#define GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH

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
  \file GGEMSRAMManager.hh

  \brief GGEMS class handling RAM memory

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday May 5, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <vector>
#include <unordered_map>

#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/global/GGEMSExport.hh"

typedef std::unordered_map<std::string, GGsize> AllocatedMemoryUMap; /*!< Unordered map with key : name of class, amount of allocated memory on OpenCL device */

/*!
  \class GGEMSRAMManager
  \brief GGEMS class handling RAM memory
*/
class GGEMS_EXPORT GGEMSRAMManager
{
  private:
    /*!
      \brief Unable the constructor for the user
    */
    GGEMSRAMManager(void);

    /*!
      \brief Unable the destructor for the user
    */
    ~GGEMSRAMManager(void);

  public:
    /*!
      \fn static GGEMSRAMManager& GetInstance(void)
      \brief Create at first time the Singleton
      \return Object of type GGEMSRAMManager
    */
    static GGEMSRAMManager& GetInstance(void)
    {
      static GGEMSRAMManager instance;
      return instance;
    }

    /*!
      \fn GGEMSRAMManager(GGEMSRAMManager const& ram_manager) = delete
      \param ram_manager - reference on the ram manager
      \brief Avoid copy of the class by reference
    */
    GGEMSRAMManager(GGEMSRAMManager const& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager& operator=(GGEMSRAMManager const& ram_manager) = delete
      \param ram_manager - reference on the ram manager
      \brief Avoid assignement of the class by reference
    */
    GGEMSRAMManager& operator=(GGEMSRAMManager const& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager(GGEMSRAMManager const&& ram_manager) = delete
      \param ram_manager - rvalue reference on the ram manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRAMManager(GGEMSRAMManager const&& ram_manager) = delete;

    /*!
      \fn GGEMSRAMManager& operator=(GGEMSRAMManager const&& ram_manager) = delete
      \param ram_manager - rvalue reference on the ram manager
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSRAMManager& operator=(GGEMSRAMManager const&& ram_manager) = delete;

    /*!
      \fn void PrintRAMStatus(void) const
      \brief print the RAM memory status for activated context
    */
    void PrintRAMStatus(void) const;

    /*!
      \fn inline bool IsEnoughAvailableRAMMemory(GGsize const& index, GGsize const& size) const
      \param index - index of device
      \param size - size in bytes to allocate
      \return true if enough available RAM memory
      \brief Checking available RAM memory on device
    */
    inline bool IsEnoughAvailableRAMMemory(GGsize const& index, GGsize const& size) const
    {
      if (size + allocated_ram_[index] < max_available_ram_[index]) return true;
      else return false;
    }

    /*!
      \fn inline bool IsBufferSizeCorrect(GGsize const& index, GGsize const& size) const
      \param index - index of device
      \param size - size in bytes of buffer
      \return true if buffer size is correct
      \brief Check the size of buffer depending on device limit, false if buffer size if too big
    */
    inline bool IsBufferSizeCorrect(GGsize const& index, GGsize const& size) const
    {
      if (size < max_buffer_size_[index]) return true;
      else return false;
    }

    /*!
      \fn void IncrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size)
      \param class_name - name of class allocating memory
      \param index - index of device
      \param size - size of the allocated buffer in byte
      \brief increment the size of the global allocated buffer
    */
    void IncrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size);

    /*!
      \fn void DecrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size)
      \param class_name - name of class deallocating memory
      \param index - index of device
      \param size - size of the allocated buffer in byte
      \brief decrement the size of the global allocated buffer
    */
    void DecrementRAMMemory(std::string const& class_name, GGsize const& index, GGsize const& size);

    /*!
      \fn void Clean(void)
      \brief clean OpenCL data if necessary
    */
    void Clean(void);

  private:
    GGsize number_detected_devices_; /*!< Number of detected device */
    GGsize* allocated_ram_; /*!< Allocated RAM on OpenCL device */
    GGsize* max_available_ram_; /*!< Max available RAM on OpenCL device */
    GGsize* max_buffer_size_; /*!< Max of buffer size of OpenCL device */
    AllocatedMemoryUMap* allocated_memories_; /*!< Allocated memory on OpenCL device by GGEMS class */
};

/*!
  \fn GGEMSRAMManager* get_instance_ggems_ram_manager(void)
  \return the pointer on the singleton
  \brief Get the GGEMSRAMManager pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSRAMManager* get_instance_ggems_ram_manager(void);

/*!
  \fn void print_infos_ram_manager(GGEMSRAMManager* ram_manager)
  \param ram_manager - pointer on the singleton
  \brief Print information about RAM memory
*/
extern "C" GGEMS_EXPORT void print_infos_ram_manager(GGEMSRAMManager* ram_manager);

#endif // End of GUARD_GGEMS_TOOLS_GGEMSRAMMANAGER_HH
