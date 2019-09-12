#include<cstdio>

struct RGB {
    unsigned char r, g, b, a;
};

extern "C" {
	__global__
	void BlackWhite(RGB *image, int width, int height) {
		int thidx = blockIdx.x*blockDim.x + threadIdx.x;

		if (thidx >= width*height) { 
			return;
		}

		unsigned char tmp = 0.299*image[thidx].r + 
		    0.587*image[thidx].g + 0.114*image[thidx].b;
		image[thidx].r = image[thidx].g = image[thidx].b = tmp;
	}

	__global__
	void Negative(RGB *oldImage, RGB *newImage, int width, int height)
    {
		int thidx = blockIdx.x*blockDim.x + threadIdx.x;
        
        if(thidx >= width*height) {
            return;
        }

        RGB tmp = oldImage[thidx];
        tmp.r = (unsigned char) 255 - tmp.r;
        tmp.g = (unsigned char) 255 - tmp.g;
        tmp.b = (unsigned char) 255 - tmp.b;

        newImage[thidx] = tmp;
    }

	__global__
	void Normalization(RGB *image, int width, int height) {
		const int thidx = blockIdx.x*blockDim.x + threadIdx.x;
		if (thidx >= width*height) { 
			return;
		}

		int tmp = image[thidx].r + image[thidx].g + image[thidx].b;
		image[thidx].r = (unsigned char) (image[thidx].r*255.0 / tmp);
		image[thidx].g = (unsigned char) (image[thidx].g*255.0 / tmp);
		image[thidx].b = (unsigned char) (image[thidx].b*255.0 / tmp);
	}
}
