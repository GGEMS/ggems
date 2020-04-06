#ifndef GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERING_HH
#define GUARD_GGEMS_PHYSICS_GGEMSCOMPTONSCATTERING_HH

/*!
  \file GGEMSComptonScattering.hh

  \brief Compton Scattering process from standard model for Geant4 (G4KleinNishinaCompton)

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 31, 2020
*/

#include "GGEMS/physics/GGEMSEMProcess.hh"

/*!
  \class GGEMSComptonScattering
  \brief Compton Scattering process from standard model for Geant4 (G4KleinNishinaCompton)
*/
class GGEMS_EXPORT GGEMSComptonScattering : public GGEMSEMProcess
{
  public:
    /*!
      \brief GGEMSComptonScattering constructor
    */
    GGEMSComptonScattering(void);

    /*!
      \brief GGEMSComptonScattering destructor
    */
    ~GGEMSComptonScattering(void);

    /*!
      \fn GGEMSComptonScattering(GGEMSComptonScattering const& compton_scattering) = delete
      \param compton_scattering - reference on the GGEMS compton scattering
      \brief Avoid copy by reference
    */
    GGEMSComptonScattering(GGEMSComptonScattering const& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering& operator=(GGEMSComptonScattering const& compton_scattering) = delete
      \param compton_scattering - reference on the GGEMS compton scattering
      \brief Avoid assignement by reference
    */
    GGEMSComptonScattering& operator=(GGEMSComptonScattering const& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering(GGEMSComptonScattering const&& compton_scattering) = delete
      \param compton_scattering - rvalue reference on the GGEMS compton scattering
      \brief Avoid copy by rvalue reference
    */
    GGEMSComptonScattering(GGEMSComptonScattering const&& compton_scattering) = delete;

    /*!
      \fn GGEMSComptonScattering& operator=(GGEMSComptonScattering const&& compton_scattering) = delete
      \param compton_scattering - rvalue reference on the GGEMS compton scattering
      \brief Avoid copy by rvalue reference
    */
    GGEMSComptonScattering& operator=(GGEMSComptonScattering const&& compton_scattering) = delete;

    /*!
      \fn void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables)
      \param particle_cross_sections - OpenCL buffer storing all the cross section tables for each particles
      \param material_tables - material tables on OpenCL device
      \brief build cross section tables and storing them in particle_cross_sections
    */
    void BuildCrossSectionTables(std::shared_ptr<cl::Buffer> particle_cross_sections, std::shared_ptr<cl::Buffer> material_tables) override;

    private:
    /*!
      \fn GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy)
      \param material_tables - activated material for a phantom
      \param material_index - index of the material
      \param energy - energy of the bin
      \return cross section for Compton for a material
      \brief compute Compton cross section for a material
    */
    GGfloat ComputeCrossSectionPerMaterial(GGEMSMaterialTables const* material_tables, GGushort const& material_index, GGfloat const& energy) override;

    /*!
      \fn GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number)
      \param energy - energy of the bin
      \param atomic_number - Z number of the chemical element
      \return Compton cross section by atom
      \brief compute Compton cross section for an atom with Klein-Nishina
    */
    GGfloat ComputeCrossSectionPerAtom(GGfloat const& energy, GGuchar const& atomic_number) override;
};

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSEMPROCESS_HH
