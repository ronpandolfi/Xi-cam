bool isOutsideBounds(const int3 pos, const int3 sizes)
{
     if(pos.x < 0 || pos.y < 0 || pos.z < 0 ||
        pos.x >= sizes.x || pos.y >= sizes.y || pos.z >= sizes.z)
        return true;
    return false;
}

int getValue(global const uchar* buffer, const int3 pos, const int3 sizes)
{
     if(pos.x < 0 || pos.y < 0 || pos.z < 0 ||
        pos.x >= sizes.x || pos.y >= sizes.y || pos.z >= sizes.z)
        return -1;

    size_t index = pos.x + pos.y*sizes.x + pos.z*sizes.x*sizes.y;

    return buffer[index];
}

void setValue(global uchar* buffer, const int3 pos, const int3 sizes, int value)
{
    size_t index = pos.x + pos.y*sizes.x + pos.z*sizes.x*sizes.y;
    buffer[index] = value;
}

void DaniMedianFilter3D(global const uchar* inputBuffer, 
                        global uchar* outputBuffer, 
                        const int3 pos, 
                        const int3 sizes, 
                        int medianIndex)
{
    int sc = 1; ///shift by 1..

	uchar median = 0;
    uchar histogram[256];
    
    for(int i = 0; i < 256; ++i)
    	histogram[i] = 0; //histogram of counts at each intensity
    
    for (int n = 0; n < 3; ++n)
    {
        for (int m = 0; m < 3; ++m)
        {
            for (int k = 0; k < 3; ++k)
            {
                int3 pos2 = { pos.x + n - sc, 
                              pos.y + m - sc, 
                              pos.z + k - sc };
                int val = getValue(inputBuffer, pos2, sizes);
                if(val >= 0) histogram[val] += 1;
            }
        }
    }

	int x = 0;
			
	median = getValue(inputBuffer, pos, sizes);

    for(int i = 0; i < 256; ++i)
    {
    	if(histogram[i] >= 0) 
    	{ 
    		x += histogram[i];
    		/// if x gets past median then we set the value..
	        median = (uchar)i;
	    	if(x >= medianIndex)
	    		break;
	   	}
    }

	//median = getValue(inputBuffer, pos, sizes);
	setValue(outputBuffer, pos, sizes, (int)median);
}

kernel void MedianFilter(global const uchar* inputBuffer,
                         global uchar* outputBuffer, 
                         int imageWidth,
                         int imageHeight, 
                         int imageDepth, 
                         int medianIndex)
{

    int3 sizes = { imageWidth, imageHeight, imageDepth };
    int3 pos = { get_global_id(0), get_global_id(1), 0 };

    if(isOutsideBounds(pos, sizes)) return;
    
    for(int i = 0; i < imageDepth; ++i)
    {
        int3 pos = { get_global_id(0), get_global_id(1), i };
        DaniMedianFilter3D(inputBuffer, outputBuffer, pos, sizes, medianIndex);
    }
}
