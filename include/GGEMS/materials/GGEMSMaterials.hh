#ifndef GUARD_GGEMS_MATERIALS_GGEMSMATERIALS_HH
#define GUARD_GGEMS_MATERIALS_GGEMSMATERIALS_HH

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
  \file GGEMSMaterials.hh

  \brief GGEMS class handling material(s) for a specific navigator

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday March 4, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include <string>

#include "GGEMS/global/GGEMSOpenCLManager.hh"
#include "GGEMS/materials/GGEMSMaterialsDatabaseManager.hh"

#include "GGEMS/materials/GGEMSMaterialsStack.hh"

class GGEMSRangeCuts;

/*!
  \class GGEMSMaterials
  \brief GGEMS class handling material(s) for a specific navigator
*/
class GGEMS_EXPORT GGEMSMaterials
{
  public:
    /*!
      \brief GGEMSMaterials constructor
    */
    GGEMSMaterials(void);

    /*!
      \brief GGEMSMaterials destructor
    */
    ~GGEMSMaterials(void);

    /*!
      \fn GGEMSMaterials(GGEMSMaterials const& materials) = delete
      \param materials - reference on the GGEMS materials
      \brief Avoid copy by reference
    */
    GGEMSMaterials(GGEMSMaterials const& materials) = delete;

    /*!
      \fn GGEMSMaterials& operator=(GGEMSMaterials const& materials) = delete
      \param materials - reference on the GGEMS materials
      \brief Avoid assignement by reference
    */
    GGEMSMaterials& operator=(GGEMSMaterials const& materials) = delete;

    /*!
      \fn GGEMSMaterials(GGEMSMaterials const&& materials) = delete
      \param materials - rvalue reference on the GGEMS materials
      \brief Avoid copy by rvalue reference
    */
    GGEMSMaterials(GGEMSMaterials const&& materials) = delete;

    /*!
      \fn GGEMSMaterials& operator=(GGEMSMaterials const&& materials) = delete
      \param materials - rvalue reference on the GGEMS materials
      \brief Avoid copy by rvalue reference
    */
    GGEMSMaterials& operator=(GGEMSMaterials const&& materials) = delete;

    /*!
      \fn void AddMaterial(std::string const& material_name)
      \param material_name - name of the material
      \brief Add a material associated to a phantom
    */
    void AddMaterial(std::string const& material_name);

    /*!
      \fn GGfloat GetDensity(std::string const& material_name) const
      \param material_name - name of the material
      \return density of material in g.cm-3
      \brief get the density of material
    */
    GGfloat GetDensity(std::string const& material_name) const;

    /*!
      \fn GGfloat GetAtomicNumberDensity(std::string const& material_name) const
      \param material_name - name of the material
      \return atomic number density of material in atom.cm-3
      \brief get the atomic number density of material
    */
    GGfloat GetAtomicNumberDensity(std::string const& material_name) const;

    /*!
      \fn GetEnergyCut(std::string const& material_name, std::string const& particle_type, GGfloat const& distance, std::string const& unit) const
      \param material_name - name of the material
      \param particle_type - type of particle
      \param distance - distance cut
      \param unit - unit of the distance
      \return energy cut in keV
      \brief Get the energy cut of material in keV
    */
    GGfloat GetEnergyCut(std::string const& material_name, std::string const& particle_type, GGfloat const& distance, std::string const& unit);

    /*!
      \fn inline std::string GetMaterialName(std::size_t i) const
      \param i - index of the material
      \return name of the material
      \brief get the name of the material at position i
    */
    inline std::string GetMaterialName(std::size_t i) const {return materials_.at(i);}

    /*!
      \fn inline ptrdiff_t GetMaterialIndex(std::string const& material_name) const
      \param material_name - name of the material
      \return index of material
      \brief get the index of the material
    */
    inline ptrdiff_t GetMaterialIndex(std::string const& material_name) const
    {
      std::vector<std::string>::const_iterator iter_mat = std::find(materials_.begin(), materials_.end(), material_name);
      if (iter_mat == materials_.end()) {
        std::ostringstream oss(std::ostringstream::out);
        oss << "Material '" << material_name << "' not found!!!" << std::endl;
        GGEMSMisc::ThrowException("GGEMSMaterials", "GetMaterialIndex", oss.str());
      }
      return std::distance(materials_.begin(), iter_mat);
    }

    /*!
      \fn inline std::size_t GetNumberOfMaterials(void) const
      \return the number of materials in the phantom
      \brief Get the number of materials in the phantom
    */
    inline std::size_t GetNumberOfMaterials(void) const {return materials_.size();}

    /*!
      \fn inline std::weak_ptr<cl::Buffer> GetMaterialTables(void) const
      \return the pointer on material tables on OpenCL device
      \brief get the pointer on material tables on OpenCL device
    */
    inline std::weak_ptr<cl::Buffer> GetMaterialTables(void) const {return material_tables_cl_;}

    /*!
      \fn iinline std::shared_ptr<GGEMSRangeCuts> GetRangeCuts(void) const
      \brief get the pointer on range cuts
      \return the pointer on range cuts
    */
    inline std::weak_ptr<GGEMSRangeCuts> GetRangeCuts(void) const {return range_cuts_cl_;}

    /*!
      \fn void SetDistanceCut(std::string const& particle_name, GGfloat const& value, std::string const& unit)
      \param particle_name - type of particle gamma, e+, e-
      \param value - value of cut
      \param unit - length unit
      \brief set the cut for a particle in distance
    */
    void SetDistanceCut(std::string const& particle_name, GGfloat const& value, std::string const& unit);

    /*!
      \fn void PrintInfos(void) const
      \brief printing labels and materials infos
    */
    void PrintInfos(void) const;

    /*!
      \fn void Initialize(void)
      \brief Initialize the materials for a navigator/phantom
    */
    void Initialize(void);

  private:
    /*!
      \fn void BuildMaterialTables(void)
      \brief Building material tables
    */
    void BuildMaterialTables(void);

  private:
    std::vector<std::string> materials_; /*!< Defined material for a phantom */
    std::shared_ptr<cl::Buffer> material_tables_cl_; /*!< Material tables on OpenCL device */
    std::shared_ptr<GGEMSRangeCuts> range_cuts_cl_; /*!< Cut for particles */
};

/*!
  \fn GGEMSMaterials* create_ggems_materials(void)
  \return the pointer on the singleton
  \brief Get the GGEMSMaterials pointer for python user.
*/
extern "C" GGEMS_EXPORT GGEMSMaterials* create_ggems_materials(void);

/*!
  \fn void add_material_ggems_materials(GGEMSMaterials* materials, char const* material_name)
  \param materials - pointer on GGEMS materials
  \param material_name - name of the material
  \brief Add a material
*/
extern "C" GGEMS_EXPORT void add_material_ggems_materials(GGEMSMaterials* materials, char const* material_name);

/*!
  \fn void initialize_ggems_materials(GGEMSMaterials* materials)
  \param materials - pointer on GGEMS materials
  \brief Intialize the tables for the materials
*/
extern "C" GGEMS_EXPORT void initialize_ggems_materials(GGEMSMaterials* materials);

/*!
  \fn void print_material_properties_ggems_materials(GGEMSMaterials* materials)
  \param materials - pointer on GGEMS materials
  \brief Print tables
*/
extern "C" GGEMS_EXPORT void print_material_properties_ggems_materials(GGEMSMaterials* materials);

/*!
  \fn GGfloat get_density_ggems_materials(GGEMSMaterials* materials, char const* material_name)
  \param materials - pointer on GGEMS materials
  \param material_name - name of the material
  \return density in g.cm-3
  \brief Get the density of material in g.cm-3
*/
extern "C" GGEMS_EXPORT GGfloat get_density_ggems_materials(GGEMSMaterials* materials, char const* material_name);

/*!
  \fn GGfloat get_energy_cut_ggems_materials(GGEMSMaterials* materials, char const* material_name, char const* particle_type, GGfloat const distance, char const* unit)
  \param materials - pointer on GGEMS materials
  \param material_name - name of the material
  \param particle_type - type of particle
  \param distance - distance cut
  \param unit - unit of the distance
  \return energy cut in keV
  \brief Get the energy cut of material in keV
*/
extern "C" GGEMS_EXPORT GGfloat get_energy_cut_ggems_materials(GGEMSMaterials* materials, char const* material_name, char const* particle_type, GGfloat const distance, char const* unit);

/*!
  \fn GGfloat get_atomic_number_density_ggems_materials(GGEMSMaterials* materials, char const* material_name)
  \param materials - pointer on GGEMS materials
  \param material_name - name of the material
  \return atomic number density in atom.cm-3
  \brief Get the density of material in g.cm-3
*/
extern "C" GGEMS_EXPORT GGfloat get_atomic_number_density_ggems_materials(GGEMSMaterials* materials, char const* material_name);

#endif // End of GUARD_GGEMS_PHYSICS_GGEMSMATERIALS_HH
