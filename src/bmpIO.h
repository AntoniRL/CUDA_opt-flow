#ifndef BMPIO_H_
#define BMPIO_H_

#include <cstdio>
#include <cstdlib>

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
            if (isGray) {
                image[(height - 1 - y) * width + x] = (float)tempImage[idx];
            } else {
                // Uśrednienie dla RGB do grayscale
                float gray = 0.299f * tempImage[idx + 2] + 0.587f * tempImage[idx + 1] + 0.114f * tempImage[idx];
                image[(height - 1 - y) * width + x] = gray;
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
    header.bfSize = sizeof(BMPHeader) + imageSize;
    header.bfOffBits = sizeof(BMPHeader);
    header.biSize = 40;
    header.biWidth = width;
    header.biHeight = height;
    header.biPlanes = 1;
    header.biBitCount = isGray ? 8 : 24;
    header.biSizeImage = imageSize;

    fwrite(&header, sizeof(BMPHeader), 1, fp);

    unsigned char *tempImage = (unsigned char *)calloc(row_padded * height, sizeof(unsigned char));

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            unsigned int idx = y * row_padded + x * (isGray ? 1 : 3);
            float val = 10.0f * image[(height - 1 - y) * width + x];
            unsigned char pixel = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);

            //unsigned char pixel = (unsigned char)image[(height - 1 - y) * width + x];
            //unsigned char pixel = (unsigned char)fminf(fmaxf(image[(height - 1 - y) * width + x], 0.0f), 255.0f);

            if (isGray) {
                tempImage[idx] = pixel;
            } else {
                tempImage[idx] = pixel;
                tempImage[idx + 1] = pixel;
                tempImage[idx + 2] = pixel;
            }
        }
    }

    fwrite(tempImage, sizeof(unsigned char), imageSize, fp);

    free(tempImage);
    fclose(fp);
}

#endif /* BMPIO_H_ */
