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

#include "GGEMS/materials/GGEMSMaterialsStack.hh"
#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/physics/GGEMSEMProcessConstants.hh"

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
      \fn void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
      \param particle_cross_sections - OpenCL buffer storing all the cross section tables for each particles
      \param material_tables - material tables on OpenCL device
      \brief build cross section tables and storing them in particle_cross_sections
    */
    virtual void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables);

  protected:
    /*!
      \fn GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
      \param material_tables - activated material for a phantom
      \param material_index - index of the material
      \param energy - energy of the bin
      \return cross section for a process for a material
      \brief compute cross section for a process for a material
    */
    GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy);

    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return cross section by atom
      \brief compute a cross section for an atom
    */
    virtual GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) const = 0;

  protected:
    GGuchar process_id_; /*!< Id of the process as defined in GGEMSEMProcessConstants.hh */
    std::string process_name_; /*!< Name of the process */
    std::string primary_particle_; /*!< Type of primary particle */
    std::string secondary_particle_; /*!< Type of secondary particle */
    bool is_secondaries_; /*!< Flag to activate secondaries */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to OpenCL manager */
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
