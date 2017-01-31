// GGEMS Copyright (C) 2017

/*!
 * \file mesh_io.cu
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Tuesday June 7, 2016
 *
 * v0.1: JB - First code
 *
 */

#ifndef MESH_IO_CU
#define MESH_IO_CU

#include "mesh_io.cuh"

/////// Main functions

MeshIO::MeshIO()
{
    m_filename   = "";
}

// Read a phasespace file
MeshData MeshIO::read_mesh_file( std::string filename )
{

    MeshData meshes;
    m_filename = filename;

    std::string ext = filename.substr( filename.find_last_of( "." ) + 1 );
    if ( ext == "raw" )
    {
        meshes = m_read_raw_data();
    }
    else if ( ext == "obj" )
    {
//        GGcout << "MeshIO: read obj mesh" << GGendl;
        meshes = m_read_obj_data();
    }
    else
    {
        GGcerr << "MeshIO can only read data in raw format (.raw) or Wavefront format (.obj)!" << GGendl;
        exit_simulation();
    }

    return meshes;
}

/////// Private functions

// Skip comment starting with "#"
void MeshIO::m_skip_comment(std::istream & is) {
    i8 c;
    i8 line[1024];
    if (is.eof()) return;
    is >> c;
    while (is && (c=='#')) {
        is.getline(line, 1024);
        is >> c;
        if (is.eof()) return;
    }
    is.unget();
}

// Read the list of tokens in a txt line
std::vector< std::string > MeshIO::m_split_txt( std::string line ) {

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));

    return tokens;

}

// Read the list of tokens in a txt line
std::vector< std::string > MeshIO::m_split_slash_txt( std::string line ) {

    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> tokens;
    char delimiter = '/';

    while ( std::getline( ss, tok, delimiter) )
    {
        tokens.push_back( tok );
    }

    return tokens;

}

// Read the list of tokens in a txt line
std::vector< std::string > MeshIO::m_split_txt_with( std::string line, i8 delimiter ) {

    std::stringstream ss(line);
    std::string tok;
    std::vector<std::string> tokens;

    while ( std::getline( ss, tok, delimiter) )
    {
        tokens.push_back( tok );
    }

    return tokens;

}

// Raw Format
// v1x v1y v1z v2x v2y v2z v3x v3y v3z

// Read data from raw data. Raw data contains only one mesh
MeshData MeshIO::m_read_raw_data()
{   

    // Open the mesh file
    std::ifstream input( m_filename.c_str(), std::ios::in );
    if( !input )
    {
        GGcerr << "Error to open the Mesh file'" << m_filename << "'!" << GGendl;
        exit_simulation();
    }

    // Compute number of triangles
    std::string line;
    ui32 N;
    while( std::getline( input, line ) ) ++N;

    // Returning to beginning of the file to read it again
    input.clear();
    input.seekg( 0, std::ios::beg );

    // Mem allocation
    MeshData mesh;
    HANDLE_ERROR( cudaMallocManaged( &(mesh.v1), N * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(mesh.v2), N * sizeof( f32xyz ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(mesh.v3), N * sizeof( f32xyz ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(mesh.mesh_index), sizeof( ui32 ) ) );
    HANDLE_ERROR( cudaMallocManaged( &(mesh.nb_triangles), sizeof( ui32 ) ) );

    HANDLE_ERROR( cudaMallocManaged( &(mesh.aabb), sizeof( AabbData ) ) );

    mesh.mesh_names.push_back( "NoName" );

    // Store data from file
    size_t idx = 0;
    f32xyz v1, v2, v3;
    f32 xmin = FLT_MAX; f32 xmax = -FLT_MAX;
    f32 ymin = FLT_MAX; f32 ymax = -FLT_MAX;
    f32 zmin = FLT_MAX; f32 zmax = -FLT_MAX;
    while( std::getline( input, line ) )
    {
        std::istringstream iss( line );
        iss >> v1.x >> v1.y >> v1.z >> v2.x >> v2.y >> v2.z >> v3.x >> v3.y >> v3.z;

        // Get min and max from every dimension (bounding box)
        if ( v1.x > xmax ) xmax = v1.x;
        if ( v2.x > xmax ) xmax = v2.x;
        if ( v3.x > xmax ) xmax = v3.x;

        if ( v1.y > ymax ) ymax = v1.y;
        if ( v2.y > ymax ) ymax = v2.y;
        if ( v3.y > ymax ) ymax = v3.y;

        if ( v1.z > zmax ) zmax = v1.z;
        if ( v2.z > zmax ) zmax = v2.z;
        if ( v3.z > zmax ) zmax = v3.z;

        if ( v1.x < xmin ) xmin = v1.x;
        if ( v2.x < xmin ) xmin = v2.x;
        if ( v3.x < xmin ) xmin = v3.x;

        if ( v1.y < ymin ) ymin = v1.y;
        if ( v2.y < ymin ) ymin = v2.y;
        if ( v3.y < ymin ) ymin = v3.y;

        if ( v1.z < zmin ) zmin = v1.z;
        if ( v2.z < zmin ) zmin = v2.z;
        if ( v3.z < zmin ) zmin = v3.z;

        // Save data
        mesh.v1[ idx ] = v1;
        mesh.v2[ idx ] = v2;
        mesh.v3[ idx ] = v3;

        ++idx;
    }

    // Save AABB
    mesh.aabb[ 0 ].xmin = xmin;
    mesh.aabb[ 0 ].xmax = xmax;
    mesh.aabb[ 0 ].ymin = ymin;
    mesh.aabb[ 0 ].ymax = ymax;
    mesh.aabb[ 0 ].zmin = zmin;
    mesh.aabb[ 0 ].zmax = zmax;

    // Close the file
    input.close();

    return mesh;

}

//  obj format (Wavefront)
//  o SolidName
//  v x y z              // Geometric vertices
//  vt x y z             // Texture coordinates
//  vn x y z             // Vertex normals
//  s off                // Smooth shading
//  f i j k              // Face indices
//  f i/a j/b k/c        // Face and texture indices
//  f i/a/u j/b/v k/c/w  // Face, texture and normal indices

// Read data from MHD format
MeshData MeshIO::m_read_obj_data()
{   

    // Open the mesh file
    std::ifstream file( m_filename.c_str(), std::ios::in );
    if( !file )
    {
        GGcerr << "Error to open the Mesh file'" << m_filename << "'!" << GGendl;
        exit_simulation();
    }

    MeshData meshes;

    std::string line;
    std::vector< std::string > keys;
    std::vector< std::string > elts;

    // Obj data
    //std::map< std::string, std::vector< f32xyz > >  vertices;
    std::vector< f32xyz > vertices;
    std::map< std::string, std::vector< ui32xyz > >  faces;

    //std::vector< f32xyz > buf_vertices;
    std::vector< ui32xyz > buf_faces;

    f32 x, y, z;
    ui32 i, j, k;
    std::string solid_name;

    // Empty the key for the beginning
    keys.clear();
    keys.push_back("");

    // Loop that read the complete file
    while ( file )
    {
        m_skip_comment( file );

        /// Search object //////////////////////////////////
        while ( keys[ 0 ] != "o" && file )
        {
            // Read a line
            std::getline( file, line );

            if ( file )
            {
                keys = m_split_txt( line );
            }
        }

        // Get the solid index ( xxx_Mesh )
        elts = m_split_txt_with( keys[ 1 ], '_' );
        solid_name = elts[ 0 ];
        meshes.mesh_names.push_back( solid_name );

        /// Then read all vertices ///////////////////////

        //buf_vertices.clear();

        // Read next line
        std::getline( file, line );
        if ( file )
        {
            keys = m_split_txt( line );
        }

        //GGcout << "Find vertices: " << keys[ 0 ] << GGendl;

        // watch dog
        if ( keys[ 0 ] != "v" )
        {
            GGcout << "Mesh file (.obj): Vertices are not stored right after the solide name!" << GGendl;
            exit_simulation();
        }

        while ( keys[ 0 ] == "v" && file )
        {
            // Read coordinates
            std::stringstream( keys[ 1 ] ) >> x;
            std::stringstream( keys[ 2 ] ) >> y;
            std::stringstream( keys[ 3 ] ) >> z;

            // Store data
            vertices.push_back( make_f32xyz( x, y, z ) );

//            if ( solid_name == "A01" )
//            {
//                GGcout << "Name: " << solid_name << " line " << line
//                       << " Find vertices: " << x << " " << y << " " << z << GGendl;
//            }

            // Read new line
            std::getline( file, line );
            if ( file )
            {
                keys = m_split_txt( line );
            }
        }

        //GGcout << "Cur line: " << line << GGendl;

        /// Searching for faces /////////////////////////

        while ( keys[ 0 ] != "f" && file )
        {
            // Read a line
            std::getline( file, line );

            if ( file )
            {
                keys = m_split_txt( line );
            }
        }

        //GGcout << "Find face: " << line << GGendl;

        /// Read all faces //////////////////////////////

        buf_faces.clear();

        // Check if faces data describe triangles and not polygons ( f x y z w )
        if ( keys.size() > 4 )
        {
            GGcerr << "Mesh file data must contains triangle mesh and not polygon mesh!" << GGendl;
            exit_simulation();
        }

        // read all faces (start with the line already readed)
        while ( keys[ 0 ] == "f" && file )
        {
            // faces can be: a/b/c or a/b or a//c or a
            // we are only interested on the first index ( vertex )
            elts = m_split_slash_txt( keys[ 1 ] );
            std::stringstream( elts[ 0 ] ) >> i;
            elts = m_split_slash_txt( keys[ 2 ] );
            std::stringstream( elts[ 0 ] ) >> j;
            elts = m_split_slash_txt( keys[ 3 ] );
            std::stringstream( elts[ 0 ] ) >> k;

            // Store data
            buf_faces.push_back( make_ui32xyz( i, j, k ) );

            //GGcout << "Find face: " << i << " " << j << " " << k << GGendl;

            // Read new line
            std::getline( file, line );
            if ( file )
            {
                keys = m_split_txt( line );
            }

        }

        // Store the complete object
        //vertices[ solid_name ] = buf_vertices;
        faces[ solid_name ] = buf_faces;

//        GGcout << "Find solid name: " << solid_name << " with " << buf_vertices.size()
//               << " vertices and " << buf_faces.size() << " faces" << GGendl;

    } // complete file


    /// Convert faces data into triangular mesh

    // First, get nb of meshes
    ui32 nb_mesh = meshes.mesh_names.size();

    // Tot nb of tri
    ui32 tot_tri = 0;
    ui32 imesh = 0;

    while ( imesh < nb_mesh )
    {
        tot_tri += faces[ meshes.mesh_names[ imesh++ ] ].size();
    }

    // Some allocations
    meshes.v1 = new f32xyz[ tot_tri ];
    meshes.v2 = new f32xyz[ tot_tri ];
    meshes.v3 = new f32xyz[ tot_tri ];
    meshes.mesh_index = new ui32[ nb_mesh ];
    meshes.nb_triangles = new ui32[ nb_mesh ];
    meshes.aabb = new AabbData[ nb_mesh ];

    // For each mesh do the convertion
    ui32 ifaces = 0;
    ui32xyz index_faces;
    ui32 nb_tri;
    ui32 gbl_offset = 0;
    ui32 gbl_index = 0;
    f32xyz v1, v2, v3;

    imesh = 0; while ( imesh < nb_mesh )
    {
        // read the name
        solid_name = meshes.mesh_names[ imesh ];

        // get the number of tri
        nb_tri = faces[ solid_name ].size();
        meshes.nb_triangles[ imesh ] = nb_tri;

        // calculate the gbl index        
        meshes.mesh_index[ imesh ] = gbl_offset;
        gbl_offset += nb_tri;

//        GGcout << "name " << solid_name << " nb tri " << nb_tri << " offset " << gbl_offset << GGendl;

        // Init AABB
        f32 xmin = FLT_MAX; f32 xmax = -FLT_MAX;
        f32 ymin = FLT_MAX; f32 ymax = -FLT_MAX;
        f32 zmin = FLT_MAX; f32 zmax = -FLT_MAX;

        // Loop over faces
        ifaces = 0; while ( ifaces < nb_tri )
        {
            // get face index
            index_faces = faces[ solid_name ][ ifaces++ ];

//            printf("        index faces %i %i %i\n ", index_faces.x, index_faces.y, index_faces.z );

            // Read vertices according faces index
            v1 = vertices[ index_faces.x - 1 ];
            v2 = vertices[ index_faces.y - 1 ];  // -1 because face index start by 1 to N and not 0 to N-1
            v3 = vertices[ index_faces.z - 1 ];

//            if ( solid_name == "B10" && imesh == 0 )
//            {

//                printf("        index faces %i %i %i   v %f %f %f - %f %f %f - %f %f %f\n ",
//                       index_faces.x, index_faces.y, index_faces.z,
//                       v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z );
//            }

            // Determine AABB
            if ( v1.x > xmax ) xmax = v1.x;
            if ( v2.x > xmax ) xmax = v2.x;
            if ( v3.x > xmax ) xmax = v3.x;

            if ( v1.y > ymax ) ymax = v1.y;
            if ( v2.y > ymax ) ymax = v2.y;
            if ( v3.y > ymax ) ymax = v3.y;

            if ( v1.z > zmax ) zmax = v1.z;
            if ( v2.z > zmax ) zmax = v2.z;
            if ( v3.z > zmax ) zmax = v3.z;

            if ( v1.x < xmin ) xmin = v1.x;
            if ( v2.x < xmin ) xmin = v2.x;
            if ( v3.x < xmin ) xmin = v3.x;

            if ( v1.y < ymin ) ymin = v1.y;
            if ( v2.y < ymin ) ymin = v2.y;
            if ( v3.y < ymin ) ymin = v3.y;

            if ( v1.z < zmin ) zmin = v1.z;
            if ( v2.z < zmin ) zmin = v2.z;
            if ( v3.z < zmin ) zmin = v3.z;

            // Store the triangle
            meshes.v1[ gbl_index ] = v1;
            meshes.v2[ gbl_index ] = v2;
            meshes.v3[ gbl_index ] = v3;

            ++gbl_index;

        }

        // Store the AABB of the corresponding mesh
        meshes.aabb[ imesh ].xmin = xmin;
        meshes.aabb[ imesh ].xmax = xmax;
        meshes.aabb[ imesh ].ymin = ymin;
        meshes.aabb[ imesh ].ymax = ymax;
        meshes.aabb[ imesh ].zmin = zmin;
        meshes.aabb[ imesh ].zmax = zmax;

        ++imesh;
    } // imesh

    return meshes;

}


//f32xyz *v1;   // Vertex 1
//f32xyz *v2;   // Vertex 2
//f32xyz *v3;   // Vertex 3

//ui32 *mesh_index;   // In case of multiple meshes
//ui32 *nb_triangles; // Nb of triangles within each mesh
//AabbData *aabb;     // Bounding box of each mesh




#endif












