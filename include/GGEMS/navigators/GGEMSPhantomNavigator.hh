#ifndef GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOMNAVIGATOR_HH
#define GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOMNAVIGATOR_HH

/*!
  \file GGEMSPhantomNavigator.hh

  \brief GGEMS mother class for phantom navigation

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

class GGEMSSolidPhantom;
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
  \class GGEMSPhantomNavigator
  \brief GGEMS mother class for phantom navigation
*/
class GGEMS_EXPORT GGEMSPhantomNavigator
{
  public:
    /*!
      \param phantom_navigator - pointer on daughter of GGEMSPhantomNavigator
      \brief GGEMSPhantomNavigator constructor
    */
    explicit GGEMSPhantomNavigator(GGEMSPhantomNavigator* phantom_navigator);

    /*!
      \brief GGEMSPhantomNavigator destructor
    */
    virtual ~GGEMSPhantomNavigator(void);

    /*!
      \fn GGEMSPhantomNavigator(GGEMSPhantomNavigator const& phantom_navigator) = delete
      \param phantom_navigator - reference on the GGEMS phantom navigator
      \brief Avoid copy by reference
    */
    GGEMSPhantomNavigator(GGEMSPhantomNavigator const& phantom_navigator) = delete;

    /*!
      \fn GGEMSPhantomNavigator& operator=(GGEMSPhantomNavigator const& phantom_navigator) = delete
      \param phantom_navigator - reference on the GGEMS phantom navigator
      \brief Avoid assignement by reference
    */
    GGEMSPhantomNavigator& operator=(GGEMSPhantomNavigator const& phantom_navigator) = delete;

    /*!
      \fn GGEMSPhantomNavigator(GGEMSPhantomNavigator const&& phantom_navigator) = delete
      \param phantom_navigator - rvalue reference on the GGEMS phantom navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantomNavigator(GGEMSPhantomNavigator const&& phantom_navigator) = delete;

    /*!
      \fn GGEMSPhantomNavigator& operator=(GGEMSPhantomNavigator const&& phantom_navigator) = delete
      \param phantom_navigator - rvalue reference on the GGEMS phantom navigator
      \brief Avoid copy by rvalue reference
    */
    GGEMSPhantomNavigator& operator=(GGEMSPhantomNavigator const&& phantom_navigator) = delete;

    /*!
      \fn void SetPhantomName(std::string const& phantom_navigator_name)
      \param phantom_navigator_name - name of the navigator
      \brief save the name of the navigator
    */
    void SetPhantomName(std::string const& phantom_navigator_name);

    /*!
      \fn void SetPhantomFile(std::string const& phantom_filename)
      \param phantom_filename - filename of MHD file for phantom
      \brief set the mhd filename for phantom
    */
    void SetPhantomFile(std::string const& phantom_filename);

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
      \fn inline std::string GetPhantomName(void) const
      \brief Get the name of the phantom
      \return the name of the phantom
    */
    inline std::string GetPhantomName(void) const {return phantom_navigator_name_;}

    /*!
      \fn inline std::shared_ptr<GGEMSSolidPhantom> GetSolidPhantom(void) const
      \brief get the pointer on solid phantom
      \return the pointer on solid phantom
    */
    inline std::shared_ptr<GGEMSSolidPhantom> GetSolidPhantom(void) const {return solid_phantom_;}

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
      \fn void PrintInfos(void) const
      \return no returned value
      \brief Printing infos about the phantom navigator
    */
    virtual void PrintInfos(void) const;

    /*!
      \fn void CheckParameters(void) const
      \return no returned value
      \brief Check mandatory parameters for a phantom
    */
    virtual void CheckParameters(void) const;

    /*!
      \fn void Initialize(void)
      \return no returned value
      \brief Initialize a GGEMS phantom
    */
    virtual void Initialize(void);

  protected:
    std::string phantom_navigator_name_; /*!< Name of the phantom navigator name */
    std::string phantom_mhd_header_filename_; /*!< Filename of MHD file for phantom */
    std::string range_data_filename_; /*!< Filename of file for range data */
    GGfloat geometry_tolerance_; /*!< Tolerance of geometry range [1mm;1nm] */
    GGfloat3 offset_xyz_; /*!< Offset of the phantom in X, Y and Z */
    bool is_offset_flag_; /*!< Apply offset */

    std::shared_ptr<GGEMSSolidPhantom> solid_phantom_; /*!< Solid phantom with geometric infos and label */
    std::shared_ptr<GGEMSMaterials> materials_; /*!< Materials of phantom */
    std::shared_ptr<GGEMSCrossSections> cross_sections_; /*!< Cross section table for process */
};

#endif // End of GUARD_GGEMS_NAVIGATORS_GGEMSPHANTOMNAVIGATOR_HH
