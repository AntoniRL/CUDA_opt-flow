#ifndef BMPIO_H_
#define BMPIO_H_

#include <cstdio>
#include <cstdlib>
#include <math.h>

#pragma pack(push, 1) // Wyrównanie do 1 bajta dla nagłówka BMP
struct BMPHeader {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;

    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};
#pragma pack(pop)

inline void getBMPSize(const char *filename, unsigned int *width, unsigned int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    BMPHeader header;
    fread(&header, sizeof(BMPHeader), 1, fp);

    if (header.bfType != 0x4D42) { // Sprawdzenie "BM" w nagłówku
        fprintf(stderr, "Invalid BMP format (must be 'BM')\n");
        exit(1);
    }

    *width = header.biWidth;
    *height = abs(header.biHeight);

    fclose(fp);
}

inline void readBMP(const char *filename, float *image, bool isGray = 0) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    BMPHeader header;
    fread(&header, sizeof(BMPHeader), 1, fp);

    if (header.bfType != 0x4D42) {
        fprintf(stderr, "Invalid BMP format\n");
        exit(1);
    }

    unsigned int width = header.biWidth;
    unsigned int height = abs(header.biHeight);
    unsigned int row_padded = ((width * (isGray ? 8 : 24) + 31) / 32) * 4;

    unsigned char *tempImage = (unsigned char *)malloc(row_padded * height);

    fseek(fp, header.bfOffBits, SEEK_SET);
    fread(tempImage, sizeof(unsigned char), row_padded * height, fp);

    // Konwersja do float*
    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int idx = y * row_padded + x * (isGray ? 1 : 3);
            if (!isGray) {
                image[(height - 1 - y) * width * 3 + x * 3] = (float)tempImage[idx + 2]; // Red
                image[(height - 1 - y) * width * 3 + x * 3 + 1] = (float)tempImage[idx + 1]; // Green
                image[(height - 1 - y) * width * 3 + x * 3 + 2] = (float)tempImage[idx]; // Blue
            }
        }
    }

    free(tempImage);
    fclose(fp);
}

void writeBMP(const char *filename, float *image, unsigned int width, unsigned int height, bool isGray = 0) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    unsigned int row_padded = ((width * (isGray ? 8 : 24) + 31) / 32) * 4;
    unsigned int imageSize = row_padded * height;

    BMPHeader header = {};
    header.bfType = 0x4D42;
    header.bfSize = sizeof(BMPHeader) + (isGray ? 256 * 4 : 0) + imageSize;
    header.bfOffBits = sizeof(BMPHeader) + (isGray ? 256 * 4 : 0);
    header.biSize = 40;
    header.biWidth = width;
    header.biHeight = height;
    header.biPlanes = 1;
    header.biBitCount = isGray ? 8 : 24;
    header.biSizeImage = imageSize;

    fwrite(&header, sizeof(BMPHeader), 1, fp);

    unsigned char *tempImage = (unsigned char *)calloc(row_padded * height, sizeof(unsigned char));
    
    if (isGray) {
        unsigned char palette[256 * 4]; // 256 kolorów, każdy 4 bajty (RGBA)
        for (int i = 0; i < 256; ++i) {
            palette[i * 4 + 0] = i; // Blue
            palette[i * 4 + 1] = i; // Green
            palette[i * 4 + 2] = i; // Red
            palette[i * 4 + 3] = 0; // Reserved
        }
        fwrite(palette, sizeof(unsigned char), 256 * 4, fp);
    }

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int idx = y * row_padded + x * (isGray ? 1 : 3);
            float val = image[(height - 1 - y) * width + x];
            unsigned char pixel = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);

            //unsigned char pixel = (unsigned char)image[(height - 1 - y) * width + x];
            //unsigned char pixel = (unsigned char)fminf(fmaxf(image[(height - 1 - y) * width + x], 0.0f), 255.0f);

            // if (isGray) {
            //     tempImage[idx] = pixel;
            // } else {
            //     tempImage[idx] = pixel;
            //     tempImage[idx + 1] = pixel;
            //     tempImage[idx + 2] = pixel;
            // }
            if (isGray) {
                tempImage[idx] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
            } else {
                tempImage[idx] = (unsigned char)fminf(fmaxf(image[(height - 1 - y) * width * 3 + x * 3 + 2], 0.0f), 255.0f); // Blue
                tempImage[idx + 1] = (unsigned char)fminf(fmaxf(image[(height - 1 - y) * width * 3 + x * 3 + 1], 0.0f), 255.0f); // Green
                tempImage[idx + 2] = (unsigned char)fminf(fmaxf(image[(height - 1 - y) * width * 3 + x * 3], 0.0f), 255.0f); // Red
            }

        }
    }

    fwrite(tempImage, sizeof(unsigned char), imageSize, fp);

    free(tempImage);
    fclose(fp);
}

#endif /* BMPIO_H_ */