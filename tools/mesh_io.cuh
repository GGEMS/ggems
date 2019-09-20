// GGEMS Copyright (C) 2017

/*!
 * \file mesh_io.cuh
 * \brief
 * \author J. Bert <bert.jul@gmail.com>
 * \version 0.1
 * \date Tuesday June 7, 2016
 *
 * v0.1: JB - First code
 *
 */

#ifndef MESH_IO_CUH
#define MESH_IO_CUH

#include "global.cuh"
#include "primitives.cuh"

struct MeshData
{
    f32xyz *v1;   // Vertex 1
    f32xyz *v2;   // Vertex 2
    f32xyz *v3;   // Vertex 3

    ui32 *mesh_index;   // In case of multiple meshes
    ui32 *nb_triangles; // Nb of triangles within each mesh
    AabbData *aabb;     // Bounding box of each mesh
    std::vector< std::string > mesh_names;     // List of the mesh names
};

// Read Mesh file
class MeshIO {

    public:
        MeshIO();
        ~MeshIO(){}

        MeshData read_mesh_file( std::string filename );

    private:
        // raw format
        MeshData m_read_raw_data();
        // obj wavefront format
        MeshData m_read_obj_data();

        void m_skip_comment( std::istream & is );
        std::vector< std::string > m_split_txt(std::string line);
        std::vector< std::string > m_split_slash_txt(std::string line);
        std::vector< std::string > m_split_txt_with( std::string line, i8 delimiter );


    private:
        std::string m_filename;






};




#endif
