#ifndef SAVE_DATA_CU
#define SAVE_DATA_CU

#include "image_reader.cuh"

#ifdef ROOT_CERN
#include "TH3D.h"
#include "TH2D.h"
#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#endif

using namespace std;


void ImageReader::create_directory(std::string dirname)
{
    std::string command = "mkdir -p " + dirname;
    system(command.c_str());
}

void ImageReader::create_directory_tree(std::string dirname)
{
    std::string tmp = dirname;;
    std::string directory = "";
    
    for(;;)
    {
        if(tmp.substr (0, tmp.find_first_of ( "/" ) +1) == "") break;
        directory += tmp.substr (0, tmp.find_first_of ( "/" ) +1);
        tmp = tmp.substr (tmp.find_first_of ( "/" ) +1 );
        
        create_directory(directory);
    }
}       

void ImageReader::record3Dimage ( string histname,  f64 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparse_compression )
{

        printf("Image Parameters : %d %g %g %d %g %g %d %g %g \n",size.x,  offset.x, offset.x+size.x*spacing.x,
                                  size.y,  offset.y, offset.y+size.y*spacing.y,
                                  size.z,  offset.z, offset.z+size.z*spacing.z );


    // Check format
    string format = get_format ( histname );
    histname = get_filename_without_format ( histname );

    create_directory_tree(histname);
    
    if ( format == "mhd" )
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";

        cout<<"Save file : "<<pathnamemhd << endl;

        // MHD file
        std::ofstream myfile;
        myfile.open ( pathnamemhd.c_str() );
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ( ( sparse_compression ) ? "COO" : "False" )  <<"\n";

        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_FLOAT\n";
        myfile << "ElementDataFile = "<<remove_path ( pathnameraw ).c_str() <<"\n";

        myfile.close();



        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen ( pathnameraw.c_str(),"wb" );

        // Compressed data in COO format
        if ( sparse_compression )
        {

            // First get the number of non-zero
            unsigned int index = 0;
            unsigned int ct_nz = 0;
            while ( index < size.x*size.y*size.z )
            {
                if ( data[index] != 0.0 ) ++ct_nz;
                ++index;
            }

            // Write the previous value as the first binary element
            fwrite ( &ct_nz, sizeof ( unsigned int ), 1, pFile_mhd );

            // Some vars
            unsigned short int ix, iy, iz;
            unsigned int jump;
            jump = size.x*size.y;
            index = 0;

            // Loop over every element
            iz = 0;
            while ( iz<size.z )
            {
                iy = 0;
                while ( iy<size.y )
                {
                    ix = 0;
                    while ( ix<size.x )
                    {

                        // Export only non-zero value in COO format
                        if ( data[index] != 0.0 )
                        {
                            // xyz coordinate
                            fwrite ( &ix, sizeof ( unsigned short int ), 1, pFile_mhd );
                            fwrite ( &iy, sizeof ( unsigned short int ), 1, pFile_mhd );
                            fwrite ( &iz, sizeof ( unsigned short int ), 1, pFile_mhd );
                            // Then the corresponding value
                            float val = data[index];
                            fwrite ( &val, sizeof ( float ), 1, pFile_mhd );
                        }

                        ++ix;
                        ++index;
                    } // ix
                    ++iy;
                } // iy
                ++iz;
            } // iz

        }
        else
        {
            // Export uncompressed raw data
            int i=0;
            while ( i<size.x*size.y*size.z )
            {
                float val = data[i];
                if ( val!=0 )
                {
                    int dx = i%size.x;
                    int dy = ( ( i - dx ) / size.x  ) %size.y;
                    int dz = ( i - dx - dy*size.x ) / ( size.x * size.y );
                    int value = dx + dy*size.x + dz * size.x * size.y;
                    i32xyz dxyz = get_bin_xyz ( i,size );
//                         printf("%d %d %d %d %d %d %d %d value %g \n",i,value,dx,dy,dz,dxyz.x,dxyz.y,dxyz.z,val);

                }

                fwrite ( &val, sizeof ( float ), 1, pFile_mhd );
                ++i;
            }
        }

        fclose ( pFile_mhd );


    } // Fin mhd
    else  if ( ( format == "ASCII" ) || ( format == "txt" ) )
    {
        string pathname = histname + ".txt";

        std::cout<<"saving "<<pathname.c_str() <<std::endl;
        std::ofstream ofs ( pathname.c_str(),  std::ofstream::out );

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for ( int i=0; i<xdim; ++i )
        {
            for ( int j=0; j<ydim; ++j )
            {
                for ( int k=0; k<zdim; ++k )
                {
                    if ( data[i + j*xdim + k*xdim*ydim]!=0. )
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }
#ifdef ROOT_CERN
// #ifndef __CUDA_ARCH__
#if defined(__cplusplus)
    else if(format == "root")
        {
        

        
        std::string pathnameroot = histname;
        pathnameroot+=".root";
        
        printf("saving %s \n",pathnameroot.c_str());
        TFile f(pathnameroot.c_str(),"recreate");

        TH3D edep("Edep", "Edep", size.x,  offset.x, offset.x+size.x*spacing.x,
                                  size.y,  offset.y, offset.y+size.y*spacing.y,
                                  size.z,  offset.z, offset.z+size.z*spacing.z );
        

        double total = 0.;
        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;


        for(int i=0; i<xdim; ++i)
            {
            for(int j=0; j<ydim; ++j)
                {
                for(int k=0; k<zdim; ++k)
                    {

                    edep.SetBinContent(i+1, (ydim-j),k+1, data[i + j*xdim + k*xdim*ydim] );

                    }
                }
            }

//         TH1D* projectionx = edep.ProjectionX("EdepX");
//         TH1D* projectiony = edep.ProjectionY("EdepY");
//         TH1D* projectionz = edep.ProjectionZ("EdepZ");
// 
//         TH1D* projectiondosex = dose.ProjectionX("DoseX");
// //         TH1D* projectiondosexmilieu = dose.ProjectionX("DoseXMilieu",ydim/2,ydim/2);
//         TH1D* projectiondosey = dose.ProjectionY("DoseY");
//         TH1D* projectiondosez = dose.ProjectionZ("DoseZ");
// // //         TH1D* projectiondosezmilieu = dose.ProjectionZ("DoseZMilieu");

        f.Write();
        f.Close();


        }
#endif
#endif
    else
    {

        cout << " Unknown format ... "<<endl;

    }

}

// void ImageReader::record3Dimage( string histname,  f32 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparse_compression )
// {
//     ImageReader::record3Dimage( histname,  (f64*)data, offset, spacing, size, sparse_compression );
//
// }
#endif