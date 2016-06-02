// GGEMS Copyright (C) 2015

/*!
 * \file vrml_io.cuh
 * \brief VRML IO
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Monday May 30, 2016
 *
 */

#ifndef VRML_IO_CU
#define VRML_IO_CU


#include "vrml_io.cuh"

// ================ Main functions ===================================

void VrmlIO::open( std::string filename )
{
    // Get ext
    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );

    // Get lower case
    std::transform( ext.begin(), ext.end(), ext.begin(), ::tolower );

    // Check extension
    if ( ext != "wrl" )
    {
        GGwarn << "GGEMS have changed the extension of the VRML file to '.wrl'" << GGendl;

        std::string filename_noext = filename.substr( 0, filename.find_last_of ( "." ) );
        filename = filename_noext + ".wrl";
    }

    // Open the file
    m_pfile = fopen( filename.c_str(), "w" );

    // Check opening
    if ( !m_pfile )
    {
        GGcerr << "Error, file " << filename << " impossible to open!" << GGendl;
        exit_simulation();
    }

    // Write the header
    fprintf( m_pfile, "#VRML V2.0 utf8\n" );
    fprintf( m_pfile, "# Exported from GGEMS\n\n" );

//    fprintf( m_pfile, "Viewpoint {\n" );
//    fprintf( m_pfile, "  position    0 0 0\n");
//    fprintf( m_pfile, "  orientation 0 0 -1 0\n");
//    fprintf( m_pfile, "  description 'view all'\n");
//    fprintf( m_pfile, "}\n\n");

    // Main Axis
    f32xyz org = { 0.0, 0.0, 0.0 };
    f32xyz ax = { 20.0, 0.0, 0.0 };
    f32xyz ay = { 0.0, 20.0, 0.0 };
    f32xyz az = { 0.0, 0.0, 20.0 };
    m_draw_axis( org, ax, ay, az );

}

void VrmlIO::close()
{
    fclose( m_pfile );
}

void VrmlIO::draw_source( BeamletSource *aSource )
{
    // Get back information from this source
    f32xyz src = aSource->get_source_origin();
    f32xyz pos = aSource->get_beamlet_position();

    m_draw_sphere( src, 10, m_yellow );
    m_draw_sphere( pos, 10, m_red );

}

void VrmlIO::draw_source( ConeBeamCTSource *aSource )
{
    // Get back information from this source
    f32xyz pos = aSource->get_position();
    f32xyz angles = aSource->get_orbiting_angles();
    f32 aperture = aSource->get_aperture();

    // TODO: when transformation will be handled in the source this section muste be removed
    TransformCalculator *T = new TransformCalculator;
    T->set_translation( pos );
    T->set_rotation( angles );
    f32matrix44 trans = T->get_transformation_matrix();
    delete T;

    m_draw_wireframe_cone( pos, aperture, trans, m_yellow );
}

void VrmlIO::draw_phantom( VoxPhanDosiNav *aPhantom )
{
    // Get back some info
    AabbData aabb = aPhantom->get_bounding_box();

    m_draw_wireframe_aabb( aabb, m_blue );
}

void VrmlIO::draw_phantom( VoxPhanImgNav *aPhantom )
{
    // Get back some info
    AabbData aabb = aPhantom->get_bounding_box();

    m_draw_wireframe_aabb( aabb, m_blue );
}

void VrmlIO::draw_detector( CTDetector *aDetector )
{
    // Get back info
    ObbData obb = aDetector->get_bounding_box();
    f32matrix44 trans = aDetector->get_transformation();

    m_draw_wireframe_obb( obb, trans, m_green );

    // local Axis
    f32xyz org = { 1.0, 1.0, 1.0 }; // Small shift of the axis for a better visibility (overlap with obb)
    f32xyz ax = { 20.0, 1.0, 1.0 };
    f32xyz ay = { 1.0, 20.0, 1.0 };
    f32xyz az = { 1.0, 1.0, 20.0 };
    org = fxyz_local_to_global_frame( trans, org );
    ax = fxyz_local_to_global_frame( trans, ax );
    ay = fxyz_local_to_global_frame( trans, ay );
    az = fxyz_local_to_global_frame( trans, az );
    m_draw_axis( org, ax, ay, az );

}

// =========== Private functions ==========================================

void VrmlIO::m_draw_point( f32xyz pos, f32xyz color )
{
    fprintf( m_pfile, "\n# Point\n" );
    fprintf( m_pfile, "Shape {\n" );
    fprintf( m_pfile, "  geometry PointSet {\n" );
    fprintf( m_pfile, "    coord Coordinate {\n" );
    fprintf( m_pfile, "      point [ %f %f %f ]\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "    color Color {\n" );
    fprintf( m_pfile, "      color [ %f %f %f ]\n", color.x, color.y, color.z );
    fprintf( m_pfile, "    }\n" );
    fprintf( m_pfile, "  }\n" );
    fprintf( m_pfile, "}\n" );
}

void VrmlIO::m_draw_sphere( f32xyz pos, f32 radius, f32xyz color, f32 transparency )
{
    fprintf( m_pfile, "\n# Sphere\n" );
    fprintf( m_pfile, "Transform {\n" );
    fprintf( m_pfile, "  translation %f %f %f\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "  children [\n" );
    fprintf( m_pfile, "     Shape {\n" );
    fprintf( m_pfile, "        geometry Sphere {\n" );
    fprintf( m_pfile, "          radius %f \n", radius );
    fprintf( m_pfile, "        }\n" );
    fprintf( m_pfile, "        appearance Appearance {\n" );
    fprintf( m_pfile, "           material Material {\n" );
    fprintf( m_pfile, "              diffuseColor %f %f %f \n", color.x, color.y, color.z );
    fprintf( m_pfile, "              transparency %f\n", transparency );
    fprintf( m_pfile, "           }\n" );
    fprintf( m_pfile, "        }\n" );
    fprintf( m_pfile, "     }\n" );
    fprintf( m_pfile, "  ]\n" );
    fprintf( m_pfile, "}\n" );
}

/* TODO: to review according to the new obb format
void VrmlIO::m_draw_obb( ObbData obb, f32xyz color, f32 transparency )
{

    fprintf( m_pfile, "\n# OBB\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", obb.translate.x, obb.translate.y, obb.translate.z );
    fprintf( m_pfile, "  rotation 1.0 0.0 0.0 %f\n", obb.angle.x /deg);  // Must be in degree
    fprintf( m_pfile, "  rotation 0.0 1.0 0.0 %f\n", obb.angle.y /deg);
    fprintf( m_pfile, "  rotation 0.0 0.0 1.0 %f\n", obb.angle.z /deg);
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Box {\n");
    fprintf( m_pfile, "        size %f %f %f\n", obb.xmax-obb.xmin, obb.ymax-obb.ymin, obb.zmax-obb.zmin);
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");

}
*/

void VrmlIO::m_draw_cone(f32xyz pos, f32xyz angles, f32 height, f32 bottom_radius, f32xyz color, f32 transparency )
{
    fprintf( m_pfile, "\n# Cone\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", pos.x, pos.y, pos.z );
    fprintf( m_pfile, "  rotation 1.0 0.0 0.0 %f\n", angles.x /deg);  // Must be in degree
    fprintf( m_pfile, "  rotation 0.0 1.0 0.0 %f\n", angles.y /deg);
    fprintf( m_pfile, "  rotation 0.0 0.0 1.0 %f\n", angles.z /deg);
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Cone {\n");
    fprintf( m_pfile, "        bottomRadius %f\n", bottom_radius);
    fprintf( m_pfile, "        height %f\n", height);
    fprintf( m_pfile, "        side TRUE\n");
    fprintf( m_pfile, "        bottom TRUE\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_aabb( AabbData aabb, f32xyz color, f32 transparency )
{
    f32 tx = 0.5f * ( aabb.xmax + aabb.xmin );
    f32 ty = 0.5f * ( aabb.ymax + aabb.ymin );
    f32 tz = 0.5f * ( aabb.zmax + aabb.zmin );

    fprintf( m_pfile, "\n# AABB\n" );
    fprintf( m_pfile, "Transform {\n");
    fprintf( m_pfile, "  translation %f %f %f\n", tx, ty, tz );
    fprintf( m_pfile, "  children [\n");
    fprintf( m_pfile, "    Shape {\n");
    fprintf( m_pfile, "      appearance Appearance {\n");
    fprintf( m_pfile, "        material Material {\n");
    fprintf( m_pfile, "          diffuseColor %f %f %f\n", color.x, color.y, color.z);
    fprintf( m_pfile, "          transparency %f\n", transparency);
    fprintf( m_pfile, "        }\n");
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "      geometry Box {\n");
    fprintf( m_pfile, "        size %f %f %f\n", aabb.xmax-aabb.xmin, aabb.ymax-aabb.ymin, aabb.zmax-aabb.zmin);
    fprintf( m_pfile, "      }\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "  ]\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_wireframe_aabb(AabbData aabb, f32xyz color )
{

    //          xmin        xmax
    //          3+---------2+
    //          /          /|
    //         /          / |
    // ymin  0+---------1+  |
    //        |          |  |
    //        | 7+       | 6+   zmax
    //        |          | /
    //        |          |/
    // ymax  4+---------5+   zmin

    fprintf( m_pfile, "\n# AABB Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymin, aabb.zmin); // 0
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymin, aabb.zmin); // 1
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymin, aabb.zmax); // 2
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymin, aabb.zmax); // 3
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymax, aabb.zmin); // 4
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymax, aabb.zmin); // 5
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmax, aabb.ymax, aabb.zmax); // 6
    fprintf( m_pfile, "        %f %f %f,\n", aabb.xmin, aabb.ymax, aabb.zmax); // 7
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");

}

void VrmlIO::m_draw_wireframe_obb( ObbData obb, f32matrix44 trans, f32xyz color )
{
    // Compute the vertice
    //          xmin        xmax
    //          3+---------2+
    //          /          /|
    //         /          / |
    // ymin  0+---------1+  |
    //        |          |  |
    //        | 7+       | 6+   zmax
    //        |          | /
    //        |          |/
    // ymax  4+---------5+   zmin

    f32xyz p0 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmin, obb.ymin, obb.zmin ) );
    f32xyz p1 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmax, obb.ymin, obb.zmin ) );
    f32xyz p2 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmax, obb.ymin, obb.zmax ) );
    f32xyz p3 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmin, obb.ymin, obb.zmax ) );
    f32xyz p4 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmin, obb.ymax, obb.zmin ) );
    f32xyz p5 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmax, obb.ymax, obb.zmin ) );
    f32xyz p6 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmax, obb.ymax, obb.zmax ) );
    f32xyz p7 = fxyz_local_to_global_frame( trans, make_f32xyz( obb.xmin, obb.ymax, obb.zmax ) );

    fprintf( m_pfile, "\n# OBB Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", p0.x, p0.y, p0.z ); // 0
    fprintf( m_pfile, "        %f %f %f,\n", p1.x, p1.y, p1.z ); // 1
    fprintf( m_pfile, "        %f %f %f,\n", p2.x, p2.y, p2.z ); // 2
    fprintf( m_pfile, "        %f %f %f,\n", p3.x, p3.y, p3.z ); // 3
    fprintf( m_pfile, "        %f %f %f,\n", p4.x, p4.y, p4.z ); // 4
    fprintf( m_pfile, "        %f %f %f,\n", p5.x, p5.y, p5.z ); // 5
    fprintf( m_pfile, "        %f %f %f,\n", p6.x, p6.y, p6.z ); // 6
    fprintf( m_pfile, "        %f %f %f,\n", p7.x, p7.y, p7.z ); // 7
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 0, 1, 2, 3, 0); // top
    fprintf( m_pfile, "      %i, %i, %i, %i, %i, -1,\n", 4, 5, 6, 7, 4); // bottom
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 4);
    fprintf( m_pfile, "      %i, %i, -1,\n", 1, 5);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 6);
    fprintf( m_pfile, "      %i, %i, -1,\n", 3, 7);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");
}

void VrmlIO::m_draw_wireframe_cone( f32xyz pos, f32 aperture, f32matrix44 trans, f32xyz color )
{
    // Compute the vertice
    //
    //                         b0 + bottom radius
    //  pos                       |
    //   +------------------------o isocenter
    //          height            |
    //                         b1 +
    //

    // Determined the major axis
    ui8 axis = 0;
    f32 max = fabs( pos.x );
    if ( fabs( pos.y ) > max )
    {
        max = fabs( pos.y );
        axis = 1;
    }
    if ( fabs( pos.z ) > max )
    {
        axis = 2;
    }

    // Get pos and height
    f32xyz zero = make_f32xyz( 0.0, 0.0, 0.0 );
    pos = fxyz_local_to_global_frame( trans, zero );
    f32 height = fxyz_mag( fxyz_sub( zero, pos ) );

    // Compute bottom radius
    f32 rad = height * tan( aperture );

    // Get b0 and b1 point
    f32xyz b0, b1;
    if ( axis == 0)
    {
        // Along z-axis
        b0 = make_f32xyz( 0.0, 0.0, -rad );
        b1 = make_f32xyz( 0.0, 0.0,  rad );
    }
    else if ( axis == 1)
    {
        // Along x-axis
        b0 = make_f32xyz( -rad, 0.0, 0.0 );
        b1 = make_f32xyz(  rad, 0.0, 0.0 );
    }
    else
    {
        // Along y-axis
        b0 = make_f32xyz( 0.0, -rad, 0.0 );
        b1 = make_f32xyz( 0.0,  rad, 0.0 );
    }


    fprintf( m_pfile, "\n# Cone Wireframe\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", pos.x, pos.y, pos.z ); // pos        0
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 0.0, 0.0 );       // isocenter  1
    fprintf( m_pfile, "        %f %f %f,\n", b0.x, b0.y, b0.z );    // b0         2
    fprintf( m_pfile, "        %f %f %f,\n", b1.x, b1.y, b1.z );    // b1         3
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 1);
    fprintf( m_pfile, "      %i, %i, -1,\n", 2, 3);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 2);
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 3);
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [%f %f %f]\n", color.x, color.y, color.z);
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 0, 0, 0]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");


}

void VrmlIO::m_draw_axis( f32xyz org, f32xyz ax, f32xyz ay, f32xyz az )
{
    fprintf( m_pfile, "\n# Axis\n" );
    fprintf( m_pfile, "Shape {\n");

    fprintf( m_pfile, "  geometry IndexedLineSet {\n");
    // Coordinate
    fprintf( m_pfile, "    coord Coordinate {\n");
    fprintf( m_pfile, "      point [\n");
    fprintf( m_pfile, "        %f %f %f,\n", org.x, org.y, org.z); // 0
    fprintf( m_pfile, "        %f %f %f,\n", ax.x, ax.y, ax.z);    // 1
    fprintf( m_pfile, "        %f %f %f,\n", ay.x, ay.y, ay.z);    // 2
    fprintf( m_pfile, "        %f %f %f,\n", az.x, az.y, az.z);    // 3
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    // CoordIndex
    fprintf( m_pfile, "    coordIndex [\n");
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 1); // X
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 2); // Y
    fprintf( m_pfile, "      %i, %i, -1,\n", 0, 3); // Z
    fprintf( m_pfile, "    ]\n");

    // Color
    fprintf( m_pfile, "    color Color {\n");
    fprintf( m_pfile, "      color [\n");
    fprintf( m_pfile, "        %f %f %f,\n", 1.0, 0.0, 0.0 ); // Red
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 1.0, 0.0 ); // Green
    fprintf( m_pfile, "        %f %f %f,\n", 0.0, 0.0, 1.0 ); // Blue
    fprintf( m_pfile, "      ]\n");
    fprintf( m_pfile, "    }\n");
    fprintf( m_pfile, "    colorIndex [0, 1, 2]\n");
    fprintf( m_pfile, "    colorPerVertex FALSE\n");

    fprintf( m_pfile, "  }\n");
    fprintf( m_pfile, "}\n");

}








#endif

















