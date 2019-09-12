#include "cuda.h"
#include "lodepng.h" // http://lodev.org/lodepng/
#include <cstdio>
#include <vector>
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

unsigned error;
void** args;
int size, threads_per_block, blocks_per_grid;

int filter = 1;  //BlackWhite 0, Negative 1, Normalization 2
const char *fName = "sample.png";

int main() {
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

    printf("loadImage(fName)\n");
    vector<unsigned char> img;
    unsigned int width, height;
    error = lodepng::decode(img, width, height, fName);
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

    printf("saveImage(\"output.png\", image)\n");
    error = lodepng_encode32_file("output.png", reinterpret_cast<const unsigned char*>(image.pixels), image.width, image.height);
    if(error != 0) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
    }
    printf("success\n");
    
    return 0;
}
