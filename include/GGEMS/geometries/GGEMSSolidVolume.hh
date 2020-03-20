#ifndef GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH
#define GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH

/*!
  \file GGEMSSolidVolume.hh

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
  \class GGEMSSolidVolume
  \brief Mother class handle solid volume
*/
class GGEMS_EXPORT GGEMSSolidVolume
{
  public:
    /*!
      \brief GGEMSSolidVolume constructor
    */
    GGEMSSolidVolume(void);

    /*!
      \brief GGEMSSolidVolume destructor
    */
    virtual ~GGEMSSolidVolume(void);

    /*!
      \fn GGEMSSolidVolume(GGEMSSolidVolume const& solid_volume) = delete
      \param solid_volume - reference on the volume solid
      \brief Avoid copy of the class by reference
    */
    GGEMSSolidVolume(GGEMSSolidVolume const& solid_volume) = delete;

    /*!
      \fn GGEMSSolidVolume& operator=(GGEMSSolidVolume const& solid_volume) = delete
      \param solid_volume - reference on the volume solid
      \brief Avoid assignement of the class by reference
    */
    GGEMSSolidVolume& operator=(GGEMSSolidVolume const& solid_volume) = delete;

    /*!
      \fn GGEMSSolidVolume(GGEMSSolidVolume const&& solid_volume) = delete
      \param solid_volume - rvalue reference on the volume solid
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSolidVolume(GGEMSSolidVolume const&& solid_volume) = delete;

    /*!
      \fn GGEMSSolidVolume& operator=(GGEMSSolidVolume const&& solid_volume) = delete
      \param solid_volume - rvalue reference on the volume solid
      \brief Avoid copy of the class by rvalue reference
    */
    GGEMSSolidVolume& operator=(GGEMSSolidVolume const&& solid_volume) = delete;

    /*!
      \fn void SetLabelValue(GGfloat const& label_value)
      \param label_value - label value in solid phantom
      \brief Set the label value
    */
    void SetLabelValue(GGfloat const& label_value);

    /*!
      \fn void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm")
      \param pos_x - position of analytical phantom in X
      \param pos_y - position of analytical phantom in Y
      \param pos_z - position of analytical phantom in Z
      \param unit - unit of the distance
      \brief Set the solid phantom position
    */
    void SetPosition(GGfloat const& pos_x, GGfloat const& pos_y, GGfloat const& pos_z, char const* unit = "mm");

    /*!
      \fn void SetMaterial(char const* material)
      \param material - name of the material
      \brief set the material, Air by default
    */
    void SetMaterial(char const* material);

    /*!
      \fn void Initialize(void)
      \return no returned value
      \brief Initialize the solid and store it in Phantom creator manager
    */
    virtual void Initialize(void) = 0;

    /*!
      \fn void Draw(void)
      \return no returned value
      \brief Draw analytical volume in voxelized phantom
    */
    virtual void Draw(void) = 0;

  protected:
    /*!
      \fn void CheckParameters(void) const
      \return no returned value
      \brief check parameters for each type of volume
    */
    virtual void CheckParameters(void) const = 0;

  protected:
    GGfloat label_value_; /*!< Value of label in solid volume */
    GGfloat3 positions_; /*!< Position of solid volume */
    std::shared_ptr<cl::Kernel> kernel_draw_solid_; /*!< Kernel drawing solid using OpenCL */
    GGEMSOpenCLManager& opencl_manager_; /*!< Reference to opencl manager singleton */
    GGEMSPhantomCreatorManager& phantom_creator_manager_; /*!< Reference to phantom creator manager */
};

#endif // End of GUARD_GGEMS_GEOMETRY_GGEMSVOLUMESOLID_HH
