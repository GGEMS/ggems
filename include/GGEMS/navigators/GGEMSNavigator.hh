#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH

/*!
  \file GGEMSNavigator.hh

  \brief GGEMS mother class for navigation

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Tuesday February 11, 2020
*/

#ifdef _MSC_VER
#pragma warning(disable: 4251) // Deleting warning exporting STL members!!!
#endif

#include "GGEMS/physics/GGEMSRangeCuts.hh"

class GGEMSSolid;
class GGEMSMaterials;
class GGEMSCrossSections;

/*!
  \namespace GGEMSTolerance
  \brief Namespace storing the tolerance for the float computations
*/
#ifndef OPENCL_COMPILER
namespace GGEMSTolerance
{
#endif
  __constant GGfloat EPSILON2 = 1.0e-02f; /*!< Epsilon of 0.01 */
  __constant GGfloat EPSILON3 = 1.0e-03f; /*!< Epsilon of 0.001 */
  __constant GGfloat EPSILON6 = 1.0e-06f; /*!< Epsilon of 0.000001 */
  __constant GGfloat GEOMETRY = 100.0f*
  #ifndef OPENCL_COMPILER
  GGEMSUnits::nm; /*!< Tolerance for the geometry navigation */
  #else
  1.e-6f; /*!< Tolerance for the geometry navigation */
  #endif
#ifndef OPENCL_COMPILER
}
#endif

/*!
  \class GGEMSNavigator
  \brief GGEMS mother class for navigator
*/
class GGEMS_EXPORT GGEMSNavigator
{
  public:
    /*!
      \param navigator - pointer on daughter of GGEMSNavigator
      \brief GGEMSNavigator constructor
    */
    explicit GGEMSNavigator(GGEMSNavigator* navigator);

    /*!
      \brief GGEMSNavigator destructor
    */
    virtual ~GGEMSNavigator(void);

    /*!
      \fn GGEMSNavigator(GGEMSNavigator const& navigator) = delete
      \param navigator - reference on the GGEMS navigator
      \brief Avoid copy by reference
    */
    GGEMSNavigator(GGEMSNavigator const& navigator) = delete;

    /*!
      \fn GGEMSNavigator& operator=(GGEMSNavigator const& navigator) = delete
      \param navigator - reference on the GGEMS navigator
      \brief Avoid assignement by reference
    */
    GGEMSNavigator& operator=(GGEMSNavigator const& navigator) = delete;

    /*!
      \fn GGEMSNavigator(GGEMSNavigator const&& navigator) = delete
      \param navigator - rvalue reference on the GGEMS navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSNavigator(GGEMSNavigator const&& navigator) = delete;

    /*!
      \fn GGEMSNavigator& operator=(GGEMSNavigator const&& navigator) = delete
      \param navigator - rvalue reference on the GGEMS navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSNavigator& operator=(GGEMSNavigator const&& navigator) = delete;

    /*!
      \fn void SetNavigatorName(std::string const& navigator_name)
      \param navigator_name - name of the navigator
      \brief save the name of the navigator
    */
    void SetNavigatorName(std::string const& navigator_name);

    /*!
      \fn void SetPhantomFile(std::string const& filename)
      \param filename - filename of MHD file for phantom
      \brief set the mhd filename for phantom
    */
    void SetPhantomFile(std::string const& filename);

    /*!
      \fn void SetRangeToMaterialFile(std::string const& range_data_filename)
      \param range_data_filename - filename with range to material data
      \brief set the range to material filename
    */
    void SetRangeToMaterialFile(std::string const& range_data_filename);

    /*!
      \fn void SetGeometryTolerance(GGfloat const& distance, std::string const& unit)
      \param distance - geometry distance
      \param unit - unit of the distance
      \brief Set the geometry tolerance in distance
    */
    void SetGeometryTolerance(GGfloat const& distance, std::string const& unit = "mm");

    /*!
      \fn void SetOffset(GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, std::string const& unit = "mm")
      \param offset_x - offset in X
      \param offset_y - offset in Y
      \param offset_z - offset in Z
      \param unit - unit of the distance
      \brief set the offset of the phantom in X, Y and Z
    */
    void SetOffset(GGfloat const offset_x, GGfloat const offset_y, GGfloat const offset_z, std::string const& unit = "mm");

    /*!
      \fn inline std::string GetNavigatorName(void) const
      \brief Get the name of the navigator
      \return the name of the navigator
    */
    inline std::string GetNavigatorName(void) const {return navigator_name_;}

    /*!
      \fn inline std::shared_ptr<GGEMSSolid> GetSolid(void) const
      \brief get the pointer on solid
      \return the pointer on solid
    */
    inline std::shared_ptr<GGEMSSolid> GetSolid(void) const {return solid_;}

    /*!
      \fn inline std::shared_ptr<GGEMSMaterials> GetMaterials(void) const
      \brief get the pointer on materials
      \return the pointer on materials
    */
    inline std::shared_ptr<GGEMSMaterials> GetMaterials(void) const {return materials_;}

    /*!
      \fn inline std::shared_ptr<GGEMSCrossSections> GetCrossSections(void) const
      \brief get the pointer on cross sections
      \return the pointer on cross sections
    */
    inline std::shared_ptr<GGEMSCrossSections> GetCrossSections(void) const {return cross_sections_;}

    /*!
      \fn void ComputeParticleNavigatorDistance(void) const
      \brief Compute distance between particle and navigator (solid)
    */
    void ComputeParticleNavigatorDistance(void) const;

    /*!
      \fn void PrintInfos(void) const
      \return no returned value
    */
    virtual void PrintInfos(void) const;

    /*!
      \fn void CheckParameters(void) const
      \return no returned value
    */
    virtual void CheckParameters(void) const;

    /*!
      \fn void Initialize(void)
      \return no returned value
    */
    virtual void Initialize(void);

  protected:
    std::string navigator_name_; /*!< Name of the navigator */
    std::string phantom_mhd_header_filename_; /*!< Filename of MHD file for phantom */
    std::string range_data_filename_; /*!< Filename of file for range data */
    GGfloat geometry_tolerance_; /*!< Tolerance of geometry range [1mm;1nm] */
    GGfloat3 offset_xyz_; /*!< Offset of the navigator in X, Y and Z */
    bool is_offset_flag_; /*!< Apply offset */

    std::shared_ptr<GGEMSSolid> solid_; /*!< Solid with geometric infos and label */
    std::shared_ptr<GGEMSMaterials> materials_; /*!< Materials of phantom */
    std::shared_ptr<GGEMSCrossSections> cross_sections_; /*!< Cross section table for process */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSNAVIGATOR_HH
