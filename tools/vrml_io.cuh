// GGEMS Copyright (C) 2015

/*!
 * \file vrml_io.cuh
 * \brief VRML IO
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday May 30, 2016
 *
 */

#ifndef VRML_IO_CUH
#define VRML_IO_CUH

#include "global.cuh"

#include "aabb.cuh"

// Sources
#include "beamlet_source.cuh"
#include "cone_beam_CT_source.cuh"

// Phantoms
#include "vox_phan_dosi_nav.cuh"

/*!
  \class VRML IO
  \brief This class allows to export GGEMS simulation object (source, phantom, etc) in a 3D representation VRML format file
*/
class VrmlIO
{
public:
    /*!
     * \brief VrmlIO contructor
     */
    VrmlIO() {}

    /*!
     * \brief VrmlIO destructor
     */
    ~VrmlIO() {}

    /*!
     * \fn void open( std::string filename )
     * \brief Open a VRML file to export a GGEMS scene
     * \param filename Name of the VRML file to export
     */
    void open( std::string filename );

    void close();

    void draw_source( BeamletSource* aSource );
    void draw_source( ConeBeamCTSource* aSource );

    void draw_phantom( VoxPhanDosiNav* aPhantom );

private:
    FILE *m_pfile;                        /*!< File to export the 3D VRML scene */

private:
    f32xyz m_yellow = {1.0, 1.0, 0.0};
    f32xyz m_red = {1.0, 0.0, 0.0};
    f32xyz m_blue = {0.0, 0.0, 1.0};
    f32xyz m_green = {0.0, 1.0, 0.0};

    void m_draw_point( f32xyz pos, f32xyz color );
    void m_draw_sphere( f32xyz pos, f32 radius, f32xyz color, f32 transparency = 0.0 );
    void m_draw_obb( ObbData obb, f32xyz color, f32 transparency = 0.0 );
    void m_draw_aabb( AabbData aabb, f32xyz color, f32 transparency = 0.0 );
    void m_draw_wireframe_aabb( AabbData aabb, f32xyz color );
    void m_draw_axis( f32xyz org, f32xyz ax, f32xyz ay, f32xyz az );

};

#endif
