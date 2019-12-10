#ifndef GUARD_GGEMS_TOOLS_MEMORY_HH
#define GUARD_GGEMS_TOOLS_MEMORY_HH

/*!
  \file memory.hh

  \brief Aligned memory

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, Brest, FRANCE
  \version 1.0
  \date Tuesday September 24, 2019
*/

#include <cstring>
#include <sstream>

#include "GGEMS/tools/print.hh"
#include "GGEMS/global/ggems_export.hh"

/*!
  \namespace Memory
  \brief namespace storing template allocating aligned memory and another
  function freeing the memory. By default the alignement is done on 128 bytes
  Usage:
  Allocating memory for 10 float with 32 bytes alignment
  float* toto = Memory::MemAlloc<float,32>( 10 );
  Freeing memory
  Memory::MemFree<float,32>( toto );
*/
namespace Memory {
  /*!
    \fn T* MemAlloc( std::size_t const& elt )
    \tparam T - Type of buffer to align
    \tparam A - Alignment in byte
    \param elt number of elements in the buffer
    \return the aligned buffer
  */
  template<typename T, std::size_t A = 128>
  T* MemAlloc(std::size_t const& elt)
  {
    std::size_t const size = sizeof(T) * elt;
    std::size_t const total_size = size + (A - 1) + sizeof(void*);

    void *p_memory = ::malloc(total_size);
    void *p_align_memory = reinterpret_cast<void*>( static_cast<uint64_t>(
      reinterpret_cast<uint64_t>(p_memory) + (A-1) + sizeof(void*))
      & ~(A-1));

    static_cast<void**>(p_align_memory)[-1] = p_memory;

    ::memset(p_align_memory, 0, size);

    // Printing a warning if the memory is not aligned
    if (reinterpret_cast<uintptr_t>(p_align_memory) % A != 0) {
      std::ostringstream oss(std::ostringstream::out);
      oss << "Address " << p_align_memory << " is not aligned on " << A
        << " bytes!!! (" << __FILE__ << ", " << __LINE__ << ")";
      throw std::runtime_error(oss.str());
    }

    return static_cast<T*>(p_align_memory);
  }

  /*!
    \fn void MemFree(void *p)
    \param p - Pointer to the aligned buffer
    \brief Free allocated memory
  */
  void GGEMS_EXPORT MemFree(void *p);
}

#endif // GUARD_GGEMS_TOOLS_MEMORY_HH
