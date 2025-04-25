/*
 * Compile: ./buildbuild
 * Run: ./blur [options]
 * Options:
 *   --input FILE    : Input image file (required)
 *   --output FILE   : Output image file (default: blurred.png)
 *   --radius N      : Set blur radius (default: 5)
 *   --verify        : Enable verification of results (default: false)
 * 
 * Example: ./blur --input image.png --output blurred.png --radius 10
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <tuple>
#include "../CU.h"

// STB Image library should be downloaded before compiling:
// curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
// curl -o stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
#include "stb_image.h"
#include "stb_image_write.h"

// ANSI color codes
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

typedef unsigned int Pixel;
typedef int INDEX;

// Simple pixel structure for RGB images
typedef struct {
    Pixel r, g, b;
} RGBPixel;

// Add two RGBPixels
__device__ __host__ RGBPixel operator+(const RGBPixel& a, const RGBPixel& b) {
    return RGBPixel{
        (Pixel)(a.r + b.r),
        (Pixel)(a.g + b.g), 
        (Pixel)(a.b + b.b)
    };
}

// Divide RGBPixel by a scalar
__device__ __host__ RGBPixel operator/(const RGBPixel& pixel, int divisor) {
    return RGBPixel{
        (Pixel)(pixel.r / divisor),
        (Pixel)(pixel.g / divisor),
        (Pixel)(pixel.b / divisor)
    };
}

__device__ __host__ INDEX get2DIndex(INDEX i, INDEX j, INDEX width) {
    return j * width + i;
}

// Placeholder for the blur kernel
__global__ void blurKernel(RGBPixel* input, RGBPixel* output, INDEX width, INDEX height, INDEX radius) {
    INDEX i = blockIdx.x * blockDim.x + threadIdx.x;
    INDEX j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds
    if (i >= width || j >= height) return;

    INDEX halfRadius = radius / 2;

    RGBPixel sum = RGBPixel{0, 0, 0}; 
    INDEX numPixels = 0; 
    for (INDEX iOffset = 0; iOffset < radius; iOffset++) {
        for (INDEX jOffset = 0; jOffset < radius; jOffset++) {
            INDEX iCur = i - halfRadius + iOffset;
            INDEX jCur = j - halfRadius + jOffset; 
            if (iCur >= 0 && iCur < width && jCur >= 0 && jCur < height) {
                sum = sum + input[get2DIndex(iCur, jCur, width)];
                numPixels += 1; 
            } 
        }
    }
    // printf("Block (%d,%d,%d), Thread %d (%d,%d,%d), Output pixel (%d,%d) = (%d,%d,%d)\n",
    //     blockIdx.x, blockIdx.y, blockIdx.z,
    //     threadIdx.x * threadIdx.y * threadIdx.z, threadIdx.x, threadIdx.y, threadIdx.z,
    //     i, j,
    //     (sum / numPixels).r,
    //     (sum / numPixels).g,
    //     (sum / numPixels).b);
    output[get2DIndex(i, j, width)] = sum / numPixels; 
}

void printUsage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --input FILE    : Input image file (required)\n");
    printf("  --output FILE   : Output image file (default: blurred.png)\n");
    printf("  --radius N      : Set blur radius (default: 5)\n");
    printf("  --verify        : Enable verification of results (default: false)\n");
    printf("\nExample: %s --input image.png --output blurred.png --radius 10\n", program_name);
}

int parseArguments(int argc, char** argv, char** input_file, char** output_file, int* radius, bool* verify) {
    // Default parameters
    *input_file = NULL;
    *output_file = (char*)"blurred.png";
    *radius = 5;
    *verify = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) {
                *input_file = argv[++i];
            } else {
                printf("Error: --input requires a filename\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                *output_file = argv[++i];
            } else {
                printf("Error: --output requires a filename\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--radius") == 0) {
            if (i + 1 < argc) {
                *radius = atoi(argv[++i]);
                if (*radius <= 0) {
                    printf("Error: Radius must be positive\n");
                    return 1;
                }
            } else {
                printf("Error: --radius requires a value\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--verify") == 0) {
            *verify = true;
        } else {
            printf("Error: Unknown option '%s'\n", argv[i]);
            return 1;
        }
    }
    
    // Check if input file was provided
    if (*input_file == NULL) {
        printf("Error: Input file is required\n");
        return 1;
    }
    
    return 0;
}

// Placeholder for loading image function
bool loadImage(const char* filename, RGBPixel** image, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 3);
    
    if (!data) {
        printf("Error: Could not load image file %s\n", filename);
        return false;
    }
    
    printf("Image loaded: %dx%d with %d channels\n", *width, *height, channels);
    
    // Allocate memory for the RGBPixel array
    *image = (RGBPixel*)malloc(*width * *height * sizeof(RGBPixel));
    if (!*image) {
        printf("Error: Failed to allocate memory for image\n");
        stbi_image_free(data);
        return false;
    }
    
    // Convert the image data to RGBPixel format
    for (int i = 0; i < *width * *height; i++) {
        (*image)[i].r = data[i * 3];
        (*image)[i].g = data[i * 3 + 1];
        (*image)[i].b = data[i * 3 + 2];
    }
    
    // Free the original image data
    stbi_image_free(data);
    
    return true;
}

// Placeholder for saving image function
bool saveImage(const char* filename, RGBPixel* image, int width, int height) {
    // Convert RGBPixel array to byte array expected by stb_image_write
    unsigned char* data = (unsigned char*)malloc(width * height * 3);
    if (!data) {
        printf("Error: Failed to allocate memory for output image\n");
        return false;
    }
    
    // Copy pixel data
    for (int i = 0; i < width * height; i++) {
        data[i * 3] = image[i].r;
        data[i * 3 + 1] = image[i].g;
        data[i * 3 + 2] = image[i].b;
    }
    
    // Determine file format based on extension
    const char* ext = strrchr(filename, '.');
    bool success = false;
    
    if (ext) {
        if (strcmp(ext, ".png") == 0) {
            success = stbi_write_png(filename, width, height, 3, data, width * 3) != 0;
        } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) {
            success = stbi_write_jpg(filename, width, height, 3, data, 95) != 0; // 95 = quality
        } else if (strcmp(ext, ".bmp") == 0) {
            success = stbi_write_bmp(filename, width, height, 3, data) != 0;
        } else {
            printf("Warning: Unrecognized file extension, saving as PNG\n");
            success = stbi_write_png(filename, width, height, 3, data, width * 3) != 0;
        }
    } else {
        printf("Warning: No file extension, saving as PNG\n");
        success = stbi_write_png(filename, width, height, 3, data, width * 3) != 0;
    }
    
    free(data);
    
    if (!success) {
        printf("Error: Failed to save image to %s\n", filename);
    }
    
    return success;
}

// Placeholder for CPU blur function (for verification)
void cpuBlur(RGBPixel* input, RGBPixel* output, int width, int height, int radius) {
    int halfRadius = radius / 2;
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            RGBPixel sum = {0, 0, 0};
            int numPixels = 0;
            
            for (int jOffset = 0; jOffset < radius; jOffset++) {
                for (int iOffset = 0; iOffset < radius; iOffset++) {
                    int iCur = i - halfRadius + iOffset;
                    int jCur = j - halfRadius + jOffset;
                    
                    if (iCur >= 0 && iCur < width && jCur >= 0 && jCur < height) {
                        sum.r += input[jCur * width + iCur].r;
                        sum.g += input[jCur * width + iCur].g;
                        sum.b += input[jCur * width + iCur].b;
                        numPixels++;
                    }
                }
            }
            
            output[j * width + i].r = sum.r / numPixels;
            output[j * width + i].g = sum.g / numPixels;
            output[j * width + i].b = sum.b / numPixels;
        }
    }
}

std::tuple<INDEX, INDEX> getMaxDimsPerBlock() {
    cudaDeviceProp p; 
    CU(cudaGetDeviceProperties(&p, 0));

    INDEX maxThreadsPerDim = sqrt(p.maxThreadsPerBlock);

    INDEX maxWidth = min(maxThreadsPerDim, p.maxThreadsDim[0]);
    INDEX maxHeight = min(maxThreadsPerDim, p.maxThreadsDim[1]);

    return std::make_tuple(maxWidth, maxHeight);
}

int main(int argc, char** argv) {
    // Parameters to be set by parseArguments
    char* input_file;
    char* output_file;
    int radius;
    bool verify;
    
    // Parse command line arguments
    if (parseArguments(argc, argv, &input_file, &output_file, &radius, &verify) != 0) {
        printUsage(argv[0]);
        return 1;
    }
    
    printf("Running with parameters:\n");
    printf("  Input file: %s\n", input_file);
    printf("  Output file: %s\n", output_file);
    printf("  Blur radius: %d\n", radius);
    printf("  Verification: %s\n", verify ? "enabled" : "disabled");
    
    // Load input image
    RGBPixel* hInputImage;
    int width, height;
    if (!loadImage(input_file, &hInputImage, &width, &height)) {
        printf("Failed to load input image\n");
        return 1;
    }
    
    printf("Image loaded: %d x %d pixels\n", width, height);
    
    // Allocate memory for output image
    RGBPixel* hOutputImage = (RGBPixel*)malloc(width * height * sizeof(RGBPixel));
    
    // Device memory allocation
    RGBPixel *dInputImage, *dOutputImage;
    CU(cudaMalloc((void**)&dInputImage, width * height * sizeof(RGBPixel)));
    CU(cudaMalloc((void**)&dOutputImage, width * height * sizeof(RGBPixel)));
    
    // Copy input image to device
    CU(cudaMemcpy(dInputImage, hInputImage, width * height * sizeof(RGBPixel), cudaMemcpyHostToDevice));

    // Get maximum dimensions for block
    auto [maxWidth, maxHeight] = getMaxDimsPerBlock();
    
    // Calculate block dimensions ensuring they don't exceed device limits
    dim3 gridDim(
        ceil(width / (float) maxWidth),
        ceil(height/ (float) maxHeight),
        1
    );
    dim3 blockDim(
        maxWidth,
        maxHeight,
        1
    );
    // Print CUDA run configuration
    printf("\nCUDA Configuration:\n");
    printf("  Grid dimensions: %d x %d x %d\n", gridDim.x, gridDim.y, gridDim.z);
    printf("  Block dimensions: %d x %d x %d\n", blockDim.x, blockDim.y, blockDim.z);
    printf("  Total threads: %d\n", gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);

    blurKernel<<<gridDim, blockDim>>>(dInputImage, dOutputImage, width, height, radius);

    CU(cudaDeviceSynchronize());
    // Copy result back to host
    CU(cudaMemcpy(hOutputImage, dOutputImage, width * height * sizeof(RGBPixel), cudaMemcpyDeviceToHost));
    
    // Verify results if requested
    if (verify) {
        printf("\nVerifying results...\n");
        
        // Allocate memory for CPU output
        RGBPixel* hCpuOutput = (RGBPixel*)malloc(width * height * sizeof(RGBPixel));
        
        // Perform CPU blur
        cpuBlur(hInputImage, hCpuOutput, width, height, radius);
        
        // Compare GPU and CPU results
        bool match = true;
        int mismatchCount = 0;
        
        for (int i = 0; i < width * height; i++) {
            if (hOutputImage[i].r != hCpuOutput[i].r ||
                hOutputImage[i].g != hCpuOutput[i].g ||
                hOutputImage[i].b != hCpuOutput[i].b) {
                
                match = false;
                mismatchCount++;
                
                // Print only the first few mismatches
                if (mismatchCount <= 5) {
                    printf("Mismatch at pixel %d: GPU(%d,%d,%d) CPU(%d,%d,%d)\n",
                        i,
                        hOutputImage[i].r, hOutputImage[i].g, hOutputImage[i].b,
                        hCpuOutput[i].r, hCpuOutput[i].g, hCpuOutput[i].b);
                }
            }
        }
        
        if (match) {
            printf(ANSI_COLOR_GREEN "Verification successful! GPU and CPU results match.\n" ANSI_COLOR_RESET);
        } else {
            printf(ANSI_COLOR_RED "Verification failed! %d%% mismatches found.\n" ANSI_COLOR_RESET, (mismatchCount * 100) / (width * height));
        }
        
        // Free CPU output memory
        free(hCpuOutput);
    }
    
    // Save output image
    if (!saveImage(output_file, hOutputImage, width, height)) {
        printf("Failed to save output image\n");
    } else {
        printf("Output image saved to %s\n", output_file);
    }
    
    // Free host memory
    free(hInputImage);
    free(hOutputImage);
    
    // Free device memory
    CU(cudaFree(dInputImage));
    CU(cudaFree(dOutputImage));
    
    printf(ANSI_COLOR_GREEN "Program completed successfully!\n" ANSI_COLOR_RESET);
    return 0;
}
