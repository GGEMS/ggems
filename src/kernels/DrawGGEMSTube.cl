#include "GGEMS/tools/GGEMSTypes.hh"

__kernel void draw_ggems_tube(
  GGdouble3 const element_sizes,
  GGuint3 const phantom_dimensions)
{
  // Getting index of thread
  GGint const kGlobalIndex = get_global_id(0);

/*
	// Take the arguments of the line
	double xCoord = 0.0, yCoord = 0.0, zCoord = 0.0, radius = 0.0, height = 0.0;
	unsigned short value = 0;

	// Scanning the line
	sscanf( line, "%*s %lf %lf %lf %lf %lf %hu", &xCoord, &yCoord, &zCoord,
		&radius, &height, &value );

	fprintf( stdout, "Drawing cylinder in %4.3f %4.3f %4.3f mm, ", xCoord, yCoord,
		zCoord );
	fprintf( stdout, "radius: %4.3f mm, height: %4.3f mm, value: %u ...\n",
		radius, height, value );

	// Taking the dimensions and sizes
	unsigned short const w = self->nX_;
	unsigned short const h = self->nY_;
	unsigned short const d = self->nZ_;

	// Size of voxels
	double const vxlSzX = self->xSize_;
	double const vxlSzY = self->ySize_;
	double const vxlSzZ = self->zSize_;

	// Conditions to be in the cylinder
	// ( x - x0 ) * ( x - x0 ) + ( y - y0 ) * ( y - y0 ) <= radius
	// -H / 2 < ( z - z0 ) < H / 2

	double const r = radius * radius;
	double const h2 = height / 2.0;

	// Take the pointer on the stack
	unsigned short* p = self->data_;

	// Loop over the whole stack
	for( unsigned short z = 0; z < d; ++z )
	{
		double Z = ( z + 0.5 ) * vxlSzZ - zCoord;
		for( unsigned short y = 0; y < h; ++y )
		{
			double Y = ( y + 0.5 ) * vxlSzY - yCoord;
			for( unsigned short x = 0; x < w; ++x )
			{
				double X = ( x + 0.5 ) * vxlSzX - xCoord;
				if( X * X + Y * Y <= r && Z >= -h2 && Z <= h2 )
				{
					*p = value;
				}
				++p; // Increment the stack
			}
		}
	}*/
}
