#include "cuda.h"
#include "lodepng.h" // http://lodev.org/lodepng/
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

struct RGB {
    unsigned char r, g, b, a;
};

class Image {
public:
    RGB *pixels;
    int width, height;

    Image(int w, int h,
        vector<unsigned char> image) : width(w), height(h) {
        pixels = new RGB[w*h];

        for (int i = 0; i < w*h; ++i) {
            pixels[i].r = image[i*4];
            pixels[i].g = image[i*4+1];
            pixels[i].b = image[i*4+2];
            pixels[i].a = image[i*4+3];
        }
    }

    ~Image() {
        delete[]pixels;
    }
};

bool validateInput(std::string sIn, int filter, std::string sOut)
{
    if(filter <= 0 || filter >= 4)
    {
        printf("Only three filters available:\n");
        printf("1 - Black and White\n");
        printf("2 - Negative\n");
        printf("3 - Normalization\n");
        return false;
    }

    if(sIn.size() < 5 || sOut.size() < 5 ||
        sIn.substr(sIn.size()-4) != ".png" ||
        sOut.substr(sOut.size()-4) != ".png")
    {
        printf("Both input and output must be in .png format\n");
        return false;
    }

    return true;
}


unsigned error;
void** args;
int size, threads_per_block, blocks_per_grid;

int main(int argc, char** argv) {
	if(argc != 4 || !validateInput(argv[1], atoi(argv[2]), argv[3]) )
    {
        printf("Usage: ./executable <srcFilename.png> <filterNumber> <dstFilename.png>\n");
        return -1;
    }

	int filter = atoi(argv[2]);
	const char *fInName = argv[1];
	const char *fOutName = argv[3];

    cuInit(0);
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if(res != CUDA_SUCCESS) { printf("Cannot acquire device 0\n"); exit(-1); }
    
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if(res != CUDA_SUCCESS) { printf("Cannot create context\n"); exit(-1); }

    CUmodule cuModule = (CUmodule) 0;
    res = cuModuleLoad(&cuModule, "filter.ptx");
    if(res != CUDA_SUCCESS) { printf("Cannot load module\n"); exit(-1); }

    printf("loadImage %s\n", fInName);
    vector<unsigned char> img;
    unsigned int width, height;
    error = lodepng::decode(img, width, height, fInName);
    if(error != 0) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
    }
    Image image = Image(width, height, img);
     
    size = image.width*image.height * sizeof(RGB);
    threads_per_block = 1024;
    blocks_per_grid = (image.height*image.width +1023)/1024;
    
    CUfunction function;
    CUdeviceptr oldImage, newImage;
    
    switch (filter) {
        case 0:
            printf("BlackWhite\n");
            res = cuModuleGetFunction(&function, cuModule, "BlackWhite");
            if(res != CUDA_SUCCESS) { printf("cannot acquire kernel handle\n"); exit(-1); }

            cuMemAlloc(&newImage, size);
            cuMemcpyHtoD(newImage, image.pixels, size);

            args = (void**)malloc(3*sizeof(void*));
            args[0]=&newImage; args[1]=&image.width; args[2]=&image.height;
            cuLaunchKernel(function, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
            break;
        case 1:
            printf("Negative\n");
            res = cuModuleGetFunction(&function, cuModule, "Negative");
            if(res != CUDA_SUCCESS) { printf("cannot acquire kernel handle\n"); exit(-1); }

            cuMemAlloc(&oldImage, size);
            cuMemAlloc(&newImage, size);
            cuMemcpyHtoD(oldImage, image.pixels, size);

            args = (void**)malloc(4*sizeof(void*));
            args[0]=&oldImage; args[1]=&newImage; args[2]=&image.width; args[3]=&image.height;
            cuLaunchKernel(function, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
            break;
        case 2:
            printf("Normalization\n");
            res = cuModuleGetFunction(&function, cuModule, "Normalization");
            if(res != CUDA_SUCCESS) { printf("cannot acquire kernel handle\n"); exit(-1); }

            cuMemAlloc(&newImage, size);
            cuMemcpyHtoD(newImage, image.pixels, size);

            args = (void**)malloc(3*sizeof(void*));
            args[0]=&newImage; args[1]=&image.width; args[2]=&image.height;
            cuLaunchKernel(function, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
            break;
    }
    cuMemcpyDtoH(image.pixels, newImage, size);

    printf("saveImage(%s, image)\n", fOutName);
    error = lodepng_encode32_file(fOutName, reinterpret_cast<const unsigned char*>(image.pixels), image.width, image.height);
    if(error != 0) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
    }
    
	printf("success\n");
    
    return 0;
}
