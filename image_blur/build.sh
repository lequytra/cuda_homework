#!/bin/bash

# Download stb_image headers if they don't exist
if [ ! -f "stb_image.h" ]; then
    echo "Downloading stb_image.h..."
    curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
fi

if [ ! -f "stb_image_write.h" ]; then
    echo "Downloading stb_image_write.h..."
    curl -o stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
fi

# Compile the STB implementation separately
echo "Compiling stb_impl.c..."
gcc -c stb_impl.c

# Compile the blur program and link with stb_impl.o
echo "Compiling blur.cu and linking..."
nvcc blur.cu stb_impl.o -o blur

echo "Build complete. Run ./blur --input <image_file> to process an image."