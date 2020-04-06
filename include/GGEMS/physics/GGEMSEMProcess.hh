#ifndef GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
#define GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH

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

#include <string>

#include "GGEMS/tools/GGEMSSystemOfUnits.hh"
#include "GGEMS/materials/GGEMSMaterialsStack.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"

/*!
  \namespace GGEMSProcessName
  \brief Namespace storing constants about processes
*/
#ifndef OPENCL_COMPILER
namespace GGEMSProcess
{
#endif
  __constant GGuchar NUMBER_PROCESSES = 1; /*!< Maximum number of processes */
  __constant GGuchar NUMBER_PHOTON_PROCESSES = 3; /*!< Maximum number of photon processes */
  //__constant GGuchar NUMBER_ELECTRON_PROCESSES = 3; /*!< Maximum number of electron processes */
  //__constant GGuchar NUMBER_PARTICLES = 5; /*!< Maximum number of different particles for secondaries */

  __constant GGuchar COMPTON_SCATTERING = 0; /*!< Compton process */
  __constant GGuchar PHOTOELECTRIC_EFFECT = 1; /*!< Photoelectric process */
  __constant GGuchar RAYLEIGH_SCATTERING = 2; /*!< Rayleigh process */
  //__constant GGuchar PHOTON_BONDARY_VOXEL = 77; /*!< Photon on the boundaries */

  //__constant GGuchar ELECTRON_IONISATION = 4; /*!< Electron ionisation process */
  //__constant GGuchar ELECTRON_MSC = 5; /*!< Electron multiple scattering process */
  //__constant GGuchar ELECTRON_BREMSSTRAHLUNG = 6; /*!< Bremsstralung electron process */

  //__constant GGuchar NO_PROCESS = 99; /*!< No process */
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \class GGEMSEMProcess
  \brief GGEMS mother class for electromagnectic process
*/
class GGEMS_EXPORT GGEMSEMProcess
{
  public:
    /*!
      \param process_name - name of the process
      \brief GGEMSEMProcess constructor
    */
    explicit GGEMSEMProcess(std::string const& process_name);

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
      \fn void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
      \param particle_cross_sections - OpenCL buffer storing all the cross section tables for each particles
      \param material_tables - material tables on OpenCL device
      \return a void pointer
      \brief build cross section tables and storing them in particle_cross_sections
    */
    virtual void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables) = 0;

  protected:
    /*!
      \fn GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
      \param material_tables - activated material for a phantom
      \param material_index - index of the material
      \param energy - energy of the bin
      \return cross section for a process for a material
      \brief compute cross section for a process for a material
    */
    virtual GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy) = 0;

    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return cross section by atom
      \brief compute a cross section for an atom
    */
    virtual GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) = 0;

  protected:
    std::string process_name_; /*!< Name of the process */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
