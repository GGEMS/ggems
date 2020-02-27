#ifndef GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATOR_HH
#define GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATOR_HH

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

#include <string>

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"

/*!
  \class GGEMSPhantomNavigator
  \brief GGEMS mother class for phantom navigation
*/
class GGEMS_EXPORT GGEMSPhantomNavigator
{
  public:
    /*!
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
      \fn void SetPhantomName(char const* phantom_navigator_name)
      \param phantom_navigator_name - name of the navigator
      \brief save the name of the navigator
    */
    void SetPhantomName(char const* phantom_navigator_name);

    /*!
      \fn void SetPhantomFile(char const* phantom_filename)
      \param phantom_filename - filename of MHD file for phantom
      \brief set the mhd filename for phantom
    */
    void SetPhantomFile(char const* phantom_filename);

    /*!
      \fn void SetRangeToMaterialFile(char const* range_data_filename)
      \param range_data_filename - filename with range to material data
      \brief set the range to material filename
    */
    void SetRangeToMaterialFile(char const* range_data_filename);

    /*!
      \fn void SetGeometryTolerance(GGdouble const& distance, char const* unit)
      \param distance - geometry distance
      \param unit - unit of the distance
      \brief Set the geometry tolerance in distance
    */
    void SetGeometryTolerance(GGdouble const& distance, char const* unit = "mm");

    /*!
      \fn void PrintInfos(void) const = 0
      \brief Printing infos about the phantom navigator
    */
    virtual void PrintInfos(void) const = 0;

    /*!
      \fn void CheckParameters(void) const
      \brief Check mandatory parameters for a phantom
    */
    virtual void CheckParameters(void) const;

    /*!
      \fn void Initialize(void)
      \brief Initialize a GGEMS phantom
    */
    virtual void Initialize(void);

  protected:
    std::string phantom_navigator_name_; /*!< Name of the phantom navigator name */
    std::string phantom_mhd_header_filename_; /*!< Filename of MHD file for phantom */
    std::string range_data_filename_; /*!< Filename of file for range data */
    GGdouble geometry_tolerance_; /*!< Tolerance of geometry range [1mm;1nm] */
};

#endif // End of GUARD_GGEMS_GEOMETRIES_GGEMSPHANTOMNAVIGATOR_HH
