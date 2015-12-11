#ifndef SAVE_DATA_CU
#define SAVE_DATA_CU

#include "image_reader.cuh"

using namespace std;



void ImageReader::record3Dimage( string histname,  f32 *data, f32xyz offset, f32xyz spacing, i32xyz size, bool sparse_compression )
{

//     cout << 

    // Check format
    string format = get_format(histname);
    histname = get_filename_without_format(histname);

    if (format == "mhd")
    {
        string pathnamemhd = histname + ".mhd";
        string pathnameraw = histname + ".raw";
        cout<<"Save file : "<<pathnamemhd << endl;

        // MHD file 
        std::ofstream myfile;
        myfile.open (pathnamemhd.c_str());
        myfile << "ObjectType = Image\n";
        myfile << "NDims = 3\n";
        myfile << "BinaryData = True\n";
        myfile << "BinaryDataByteOrderMSB = False\n";

        myfile << "CompressedData = "<<  ((sparse_compression) ? "COO" : "False")  <<"\n";
        
        myfile << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
        myfile << "Offset = "<<offset.x<<" "<<offset.y<<" "<<offset.z<<"\n";
        myfile << "CenterOfRotation = 0 0 0\n";
        myfile << "ElementSpacing = "<<spacing.x<<" "<<spacing.y<<" "<<spacing.z<<"\n";
        myfile << "DimSize = "<<size.x<<" "<<size.y<<" "<<size.z<<"\n";
        myfile << "AnatomicalOrientation = ???\n";
        myfile << "ElementType = MET_FLOAT\n";
        myfile << "ElementDataFile = "<<pathnameraw.c_str()<<"\n";

        myfile.close();



        // RAW File
        FILE *pFile_mhd;
        pFile_mhd = fopen(pathnameraw.c_str(),"wb");

            // Compressed data in COO format
            if ( sparse_compression ) {

                // First get the number of non-zero
                unsigned int index = 0;
                unsigned int ct_nz = 0;
                while (index < size.x*size.y*size.z) {
                    if (data[index] != 0.0) ++ct_nz;
                    ++index;
                }

                // Write the previous value as the first binary element
                fwrite(&ct_nz, sizeof(unsigned int), 1, pFile_mhd);

                // Some vars
                unsigned short int ix, iy, iz;
                unsigned int jump;
                jump = size.x*size.y;
                index = 0;

                // Loop over every element
                iz = 0; while (iz<size.z) {
                    iy = 0; while (iy<size.y) {
                        ix = 0; while (ix<size.x) {

                            // Export only non-zero value in COO format
                            if (data[index] != 0.0) {
                                // xyz coordinate
                                fwrite(&ix, sizeof(unsigned short int), 1, pFile_mhd);
                                fwrite(&iy, sizeof(unsigned short int), 1, pFile_mhd);
                                fwrite(&iz, sizeof(unsigned short int), 1, pFile_mhd);
                                // Then the corresponding value
                                float val = data[index];
                                fwrite(&val, sizeof(float), 1, pFile_mhd);
                            }

                            ++ix; ++index;
                        } // ix
                        ++iy;
                    } // iy
                    ++iz;
                } // iz

            } else {
                // Export uncompressed raw data
                int i=0; while (i<size.x*size.y*size.z) {
                    float val = data[i]; 
                    fwrite(&val, sizeof(float), 1, pFile_mhd);
                    ++i;
                }
            }

        fclose(pFile_mhd);

        
    } // Fin mhd
#ifdef ROOT
    else if (format == "root")
    {
        string pathname = histname + ".root";
    
        cout<<"Save file : "<<pathname << endl;

    
    }
#endif
    else  if ((format == "ASCII") || (format == "txt"))
    {
        string pathname = histname + ".txt";

        std::cout<<"saving "<<pathname.c_str()<<std::endl;
        std::ofstream ofs (pathname.c_str(),  std::ofstream::out);

        int xdim = size.x;
        int ydim = size.y;
        int zdim = size.z;

        for(int i=0; i<xdim; ++i)
        {
            for(int j=0; j<ydim; ++j)
            {
                for(int k=0; k<zdim; ++k)
                {
                    if(data[i + j*xdim + k*xdim*ydim]!=0.)
                        ofs << i <<"\t"<<j<<"\t"<<k<<"\t"<<data[i + j*xdim + k*xdim*ydim]<<std::endl;
                }
            }
        }
    }
    else
    {
    
    cout << " Unknown format ... "<<endl;
    
    }

}


#endif 