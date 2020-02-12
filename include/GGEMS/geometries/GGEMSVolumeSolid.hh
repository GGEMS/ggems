#ifndef GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH
#define GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH

/*!
  \file GGEMSVolumeSolid.hh

  \brief Mother class handle solid volume

  \author Julien BERT <julien.bert@univ-brest.fr>
  \author Didier BENOIT <didier.benoit@inserm.fr>
  \author LaTIM, INSERM - U1101, Brest, FRANCE
  \version 1.0
  \date Monday January 13, 2020
*/

#include "GGEMS/global/GGEMSExport.hh"
#include "GGEMS/tools/GGEMSTypes.hh"
#include "GGEMS/geometries/GGEMSPhantomCreatorManager.hh"

/*!
  \class GGEMSVolumeSolid
  \brief Mother class handle solid volume
*/
class GGEMS_EXPORT GGEMSVolumeSolid
{
  public:
    /*!
      \brief GGEMSVolumeSolid constructor
    */
    GGEMSVolumeSolid(void);

    /*!
      \brief GGEMSVolumeSolid destructor
    */
    virtual ~GGEMSVolumeSolid(void);

  public:
    /*!
      \fn GGEMSVolumeSolid(GGEMSVolumeSolid const& volume_solid) = delete
      \param volume_solid - reference on the volume solid
      \brief Avoid copy of the class by reference
    */
    GGEMSVolumeSolid(GGEMSVolumeSolid const& volume_solid) = delete;

    /*!
      \fn GGEMSVolumeSolid& operator=(GGEMSVolumeSolid const& volume_solid) = delete
      \param volume_solid - reference on the volume solid
      \brief Avoid assignement of the class by reference
    */
    GGEMSVolumeSolid& operator=(GGEMSVolumeSolid const& volume_solid) = delete;

    /*!
      \fn GGEMSVolumeSolid(GGEMSVolumeSolid const&& volume_solid) = delete
      \param volume_solid - rvalue reference on the volume solid
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSVolumeSolid(GGEMSVolumeSolid const&& volume_solid) = delete;

    /*!
      \fn GGEMSVolumeSolid& operator=(GGEMSVolumeSolid const&& volume_solid) = delete
      \param volume_solid - rvalue reference on the volume solid
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSVolumeSolid& operator=(GGEMSVolumeSolid const&& volume_solid) = delete;

  public:
    /*!
      \fn void SetLabelValue(GGfloat const& label_value)
      \param label_value - label value in solid phantom
      \brief Set the label value
    */
    void SetLabelValue(GGfloat const& label_value);

    /*!
      \fn void SetPosition(GGdouble const& pos_x, GGdouble const& pos_y, GGdouble const& pos_z)
      \param pos_x - position of analytical phantom in X
      \param pos_y - position of analytical phantom in Y
      \param pos_z - position of analytical phantom in Z
      \brief Set the solid phantom position
    */
    void SetPosition(GGdouble const& pos_x, GGdouble const& pos_y,
      GGdouble const& pos_z);

  public:
    /*!
      \fn void Initialize(void)
      \brief Initialize the solid and store it in Phantom creator manager
    */
    virtual void Initialize(void) = 0;

    /*!
      \fn void Draw(void)
      \brief Draw analytical volume in voxelized phantom
    */
    virtual void Draw(void) = 0;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \brief check parameters for each type of volume
    */
    virtual void CheckParameters(void) const = 0;

  protected:
    GGfloat label_value_; /*!< Value of label in solid volume */
    GGdouble3 positions_; /*!< Position of solid volume */

  protected: // kernel draw solid
    cl::Kernel* p_kernel_draw_solid_; /*!< Kernel drawing solid using OpenCL */

  protected:
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
    GGEMSPhantomCreatorManager& phantom_creator_manager_; /*!< Reference to phantom creator manager */
};

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH
