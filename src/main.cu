#include "bmpIO.h"
#include "ppmIO.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string>

#define TILE_SIZE 16

__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        // get 1D coordinate for the grayscale image
        int grayOffset = y * width + x;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset * channels;
        float r = rgbImage[rgbOffset];     // red value for pixel
        float g = rgbImage[rgbOffset + 1]; // green value for pixel
        float b = rgbImage[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

/*
* Kernel do konwersji przepływu optycznego na obraz RGB.
*/
__global__ void  FlowToRGB_Kernel(float* d_u, 
                                  float* d_v, 
                                  float* d_O, 
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float u = d_u[idx];
        float v = d_v[idx];

        // Normalizacja przepływu
        float magnitude = sqrtf(u * u + v * v);
        float angle = -atan2f(v, u);

        float norm_maggnitude = magnitude / 4;
        float norm_angle = (angle + M_PI) / (2.0f * M_PI);
        
        float H = norm_angle*360.0f;
        float S = norm_maggnitude;
        float V = 1.0f;
        
        float r, g, b;
        float C = V * S;
        float X = C * (1 - fabsf(fmodf(H / 60.0f, 2) - 1));
        float m = V - C;

        if (H >= 0 && H < 60) {
            r = C; g = X; b = 0;
        } else if (H >= 60 && H < 120) {
            r = X; g = C; b = 0;
        } else if (H >= 120 && H < 180) {
            r = 0; g = C; b = X;
        } else if (H >= 180 && H < 240) {
            r = 0; g = X; b = C;
        } else if (H >= 240 && H < 300) {
            r = X; g = 0; b = C;
        } else {
            r = C; g = 0; b = X;
        }

        r = (r+m) * 255.0f;
        g = (g+m) * 255.0f;
        b = (b+m) * 255.0f;
        
        // Przypisanie wartości do tablicy wyjściowej
        d_O[3 * idx + 0] = r;
        d_O[3 * idx + 1] = g;
        d_O[3 * idx + 2] = b;
    }
}

/**
 * Kernel Lucas–Kanade (bardzo uproszczony):
 * Każdy piksel liczy sumy w oknie [x-r, x+r] x [y-r, y+r], 
 * rozwiązuje układ 2x2 (o ile wyznacznik != 0).
 *
 * dIx, dIy, dIt: gradienty obliczone wcześniej
 * dU, dV       : tu zapisujemy wynikowy wektor przepływu (u,v)
 * width, height
 */
__global__ void LukasKanadeKernel(const float* d_N1,
                                  const float* d_N2,
                                  float* d_u,
                                  float* d_v,
                                  int width, int height,
                                  int r) // promień okienka
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO : Pamięć współdzielona

    // Pomijamy piksele na brzegu
    if (x < r+1 || x >= width - r+1) return;
    if (y < r+1 || y >= height - r+1) return;

    // liczymy gradient dla piksela (Matlab metoda 'central')
    int idx = y * width + x;

    // Macierz A i wektor B
    float A11 = 0.0f, A12_21 = 0.0f, A22 = 0.0f;
    float B1  = 0.0f, B2  = 0.0f;

    // Sumowanie w małym oknie
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            // liczymy gradient dla piksela (Matlab metoda 'central')
            int current_idx = (y + dy) * width + (x + dx);

            float Ix = (d_N1[current_idx + 1] - d_N1[current_idx - 1]) * 0.5f;
            float Iy = (d_N1[current_idx + width] - d_N1[current_idx - width]) * 0.5f;
            float It = d_N2[current_idx] - d_N1[current_idx];

            // Macierz A = sum( [ix^2  ix*iy ; ix*iy  iy^2 ] )
            A11 += Ix*Ix;
            A12_21 += Ix*Iy;
            A22 += Iy*Iy;

            // Wektor B = - sum( [ix*it ; iy*it] )
            B1 -= Ix*It;
            B2 -= Iy*It;
        }
    }

    // Wyznacznik
    float u = 0.0f;
    float v = 0.0f;
    float det = A11 * A22 - A12_21 * A12_21;
    if (fabs(det) > 1e-6f) {
        // Rozwiązanie układu 2x2
        u = (A22 * B1 - A12_21 * B2) / det;
        v = (A11 * B2 - A12_21 * B1) / det;
    } else {
        // Jeśli macierz prawie osobliwa, u = v = 0
        u = 0.0f;
        v = 0.0f;
    }

    d_u[y * width + x] = u;
    d_v[y * width + x] = v;
}

__global__ void hornSchunckKernel(const float* d_frame1, // pierwsza klatka
                                           const float* d_frame2, // druga klatka
                                           float* d_u, // komponenty prędkości w kierunku x
                                           float* d_v, // komponenty prędkości w kierunku y
                                           int width, int height, // wymiary obrazu
                                           float alpha, // współczynnik gładkości
                                           int numIterations) // liczba iteracji
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Sprawdzamy, czy indeksy mieszczą się w obrębie obrazu
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) return;

    // Indeksy do pikseli w obrazach
    int idx = y * width + x;
    int idxRight = y * width + (x + 1); // Piksel po prawej stronie
    int idxLeft = y * width + (x - 1); // Piksel po lewej stronie
    int idxDown = (y + 1) * width + x; // Piksel poniżej
    int idxUp = (y - 1) * width + x; // Piksel powyżej
    int idxNextFrame = (y * width + x); // Piksel w tej samej pozycji w drugiej klatce

    // Obliczanie gradientów przestrzennych (I_x, I_y)
    float I_x = (d_frame1[idxRight] - d_frame1[idxLeft]) * 0.5f;
    float I_y = (d_frame1[idxDown] - d_frame1[idxUp]) * 0.5f;

    // Obliczanie gradientu czasowego (I_t)
    float I_t = d_frame2[idxNextFrame] - d_frame1[idxNextFrame];

    // Obliczanie średnich prędkości (u, v) z poprzednich kroków (na początek 0)
    float u = d_u[idx];
    float v = d_v[idx];

    for (int iter = 0; iter < numIterations; ++iter)
    {
        // Obliczanie różnicy między przepływem a gradientem
        float num_u = I_x * (I_x * u + I_y * v + I_t);
        float num_v = I_y * (I_x * u + I_y * v + I_t);

        // Obliczanie mianowników (z odpowiednimi rozważeniami o sąsiednich gradientach)
        float den = I_x * I_x + I_y * I_y + alpha;

        // Nowe wartości przepływu
        u = u - (num_u / den);
        v = v - (num_v / den);
    }

    // Przypisanie nowych wartości przepływu do wynikowego obrazu
    d_u[idx] = u;
    d_v[idx] = v;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("Użycie: %s <obraz1.bmp> <obraz2.bmp> <metoda=LK lub HS>\n", argv[0]);
        return 0;
    }

    const char* file1 = argv[1];
    const char* file2 = argv[2];
    const char* output_file = argv[3];
    std::string method = argv[4];

    // --- Wczytanie bitmap
    unsigned int width, height;
    getPPMSize(file1, &width, &height);

    float* h_N1 = (float*)malloc(width * height * sizeof(float));
    float* h_N2 = (float*)malloc(width * height * sizeof(float));

    readPPM(file1, h_N1, /*isGray=*/true);
    readPPM(file2, h_N2, /*isGray=*/true);

    printf("Wczytano plik1: %s\n", file1);
    printf("Wczytano plik2: %s\n", file2);
    printf("width = %d, height = %d\n", width, height);

    // --- Alokacja GPU
    float *d_N1, *d_N2;
    cudaMalloc((void**)&d_N1,  width * height * sizeof(float));
    cudaMalloc((void**)&d_N2,  width * height * sizeof(float));

    cudaMemcpy(d_N1, h_N1, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N2, h_N2, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Przygotowanie tablic na wyniki (U, V)
    float *d_u, *d_v; 
    cudaMalloc((void**)&d_u, width * height * sizeof(float));
    cudaMalloc((void**)&d_v, width * height * sizeof(float));
    cudaMemset(d_u, 0, width * height * sizeof(float));
    cudaMemset(d_v, 0, width * height * sizeof(float));

    // Hostowe tablice na wynik przepływu
    float* h_u = (float*)malloc(width * height * sizeof(float));
    float* h_v = (float*)malloc(width * height * sizeof(float));

    if (method == "LK") {
        printf("=== Lucas–Kanade ===\n");
        // Ustawmy promień okna (np. 4 => okno 9x9)
        int r = 4;

        dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
        dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
        LukasKanadeKernel<<<dimGrid, dimBlock>>>(d_N1, d_N2, d_u, d_v, width, height, r);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

    } else if (method == "HS") {
        printf("=== Horn–Schunck ===\n");

        // Współczynnik gładkości
        float alfa = 20.0f;
        int numIters = 1;

        dim3 dimBlock_2(TILE_SIZE, TILE_SIZE, 1);
        dim3 dimGrid_2((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
        hornSchunckKernel<<<dimGrid_2, dimBlock_2>>>(d_N1, d_N2, d_u, d_v, width, height, alfa, numIters);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

    } else {
        printf("Metoda nie rozpoznana! Użyj 'LK' lub 'HS'.\n");
        return 0;
    }

    // Konwersja przepływu do RGB
    float *d_O;
    cudaMalloc((void**)&d_O,  3 * width * height * sizeof(float));

    dim3 dimBlock_1(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid_1((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
    FlowToRGB_Kernel<<<dimGrid_1, dimBlock_1>>>(d_u, d_v, d_O, width, height);
    cudaDeviceSynchronize();

    // Kopiowanie wyników z GPU
    float* h_O = (float*)malloc(3 * width * height * sizeof(float));
    cudaMemcpy(h_O, d_O, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Zapis do BMP
    try{
        writePPM(output_file, h_O, width, height, false);
        printf("Zapisano plik: %s\n", output_file);
    } catch (...) {
        printf("Nie udało się zapisać pliku flow.bmp\n");    
    }

    // Zwolnienie pamięci
    free(h_N1);
    free(h_N2);
    free(h_u);
    free(h_v);
    free(h_O);

    cudaFree(d_N1);
    cudaFree(d_N2);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_O);

    return 0;
}

