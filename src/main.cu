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
        float mag = sqrtf(u * u + v * v);
        float angle = atan2f(v, u);

        float normalized_angle = (angle + M_PI) / (2.0f * M_PI);

        float S = 1.0f, V = 1.0f;
        float H = normalized_angle * 360.0f;

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

        r += m * 255.0f;
        g += m * 255.0f;
        b += m * 255.0f;
        
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
    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

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
            float Iy = (d_N1[current_idx + width] - d_N1[current_idx - 1]) * 0.5f;
            float It = d_N2[idx] - d_N1[idx];

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
    float det = A11 * A22 - A12_21 * A12_21;
    if (fabs(det) > 1e-6f) {
        // Rozwiązanie układu 2x2
        float u = (A22 * B1 - A12_21 * B2) / det;
        float v = (A11 * B2 - A12_21 * B1) / det;
    } else {
        // Jeśli macierz prawie osobliwa, u = v = 0
        v = 0.0f;
        u = 0.0f;
    }

    d_u[y * width + x] = u;
    d_v[y * width + x] = v;
}

/**
 * Kernel Horn–Schunck – jedna iteracja.
 * dIx, dIy, dIt : gradienty
 * dU, dV       : przepływ z poprzedniej iteracji (także aktualizowany)
 * dUout, dVout : przepływ wyliczony w bieżącej iteracji
 * alpha        : współczynnik regular. (im większy, tym bardziej gładki przepływ)
 */
__global__ void hornSchunckIteration(const float* __restrict__ dIx,
                                     const float* __restrict__ dIy,
                                     const float* __restrict__ dIt,
                                     const float* __restrict__ dU,
                                     const float* __restrict__ dV,
                                     float* __restrict__ dUout,
                                     float* __restrict__ dVout,
                                     int width, int height,
                                     float alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= width - 1) return;
    if (y < 1 || y >= height - 1) return;

    int idx = y * width + x;

    float ix = dIx[idx];
    float iy = dIy[idx];
    float it = dIt[idx];

    // Średnie U, V z 4 sąsiadów (góra, dół, lewo, prawo) + bieżący
    float uN = dU[(y-1) * width + x];
    float uS = dU[(y+1) * width + x];
    float uW = dU[y * width + (x-1)];
    float uE = dU[y * width + (x+1)];

    float vN = dV[(y-1) * width + x];
    float vS = dV[(y+1) * width + x];
    float vW = dV[y * width + (x-1)];
    float vE = dV[y * width + (x+1)];

    // Średnia sąsiadów
    float uAvg = 0.25f * (uN + uS + uW + uE);
    float vAvg = 0.25f * (vN + vS + vW + vE);

    // Liczymy wspólny czynnik (Ix*uAvg + Iy*vAvg + It)
    float t = (ix * uAvg + iy * vAvg + it);
    float denom = alpha * alpha + ix * ix + iy * iy; // alpha^2 + Ix^2 + Iy^2

    // Uaktualniamy nowy U, V
    float uNew = uAvg - (ix * t) / denom;
    float vNew = vAvg - (iy * t) / denom;

    dUout[idx] = uNew;
    dVout[idx] = vNew;
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
    getBMPSize(file1, &width, &height);

    float* h_N1 = (float*)malloc(width * height * sizeof(float));
    float* h_N2 = (float*)malloc(width * height * sizeof(float));

    readBMP(file1, h_N1, /*isGray=*/true);
    readBMP(file2, h_N2, /*isGray=*/true);

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

    } else if (method == "HS") {
        printf("=== Horn–Schunck ===\n");

        // TODO : Horn–Schunck

    } else {
        printf("Metoda nie rozpoznana! Użyj 'LK' lub 'HS'.\n");
        return 0;
    }

    // Konwersja przepływu do RGB
    float *d_O;
    cudaMalloc((void**)&d_O,  3 * width * height * sizeof(float));

    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
    FlowToRGB_Kernel<<<dimGrid, dimBlock>>>(d_u, d_v, d_O, width, height);
    cudaDeviceSynchronize();

    // Kopiowanie wyników z GPU
    float* h_O = (float*)malloc(3 * width * height * sizeof(float));
    cudaMemcpy(h_O, d_O, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Zapis do BMP
    try:
        writeBMP(output_file, h_O, width, height, false);
        printf("Zapisano plik flow.bmp\n");
    except:
        printf("Nie udało się zapisać pliku flow.bmp\n");    

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


// __global__ void optFlow_LucasKanade(float *N1, float *N2, float *P, const int height, const int width)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
//     {
//         P[y * width + x] = 0.0f;
//         return;
//     }

//     // Obliczenie gradientów
//     float Ix = (N1[y * width + (x + 1)] - N1[y * width + (x - 1)]) / 2.0f;
//     float Iy = (N1[(y + 1) * width + x] - N1[(y - 1) * width + x]) / 2.0f;
//     float It = N2[y * width + x] - N1[y * width + x];

//     // Dodanie zabezpieczenia przed dzieleniem przez zero
//     float flow_u = 0.0f;
//     float flow_v = 0.0f;

//     float flow_magnitude = hypotf(flow_u, flow_v);

//     // Debugowanie dla pikseli
//     printf("Ix = %f, Iy = %f, It = %f, flow_u = %f, flow_v = %f, flow_magnitude = %f\n", 
//             Ix, Iy, It, flow_u, flow_v, flow_magnitude);
// }





// __global__ void optFlow_HornSchunck(float *N1, float *N2, float *P, const int height, const int width)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     // Sprawdzenie granic
//     if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1)
//     {
//         P[y * width + x] = 0.0f;
//         return;
//     }

//     // Gradienty przestrzenne i czasowe
//     float Ix = (N1[y * width + (x + 1)] - N1[y * width + (x - 1)]) / 2.0f;
//     float Iy = (N1[(y + 1) * width + x] - N1[(y - 1) * width + x]) / 2.0f;
//     float It = N2[y * width + x] - N1[y * width + x];

//     // Obliczanie średniej przepływu optycznego (Horn-Schunck)
//     float avg_u = 0.0f;
//     int count = 0;

//     if (x > 0) { avg_u += P[y * width + (x - 1)]; count++; }
//     if (x < width - 1) { avg_u += P[y * width + (x + 1)]; count++; }
//     if (y > 0) { avg_u += P[(y - 1) * width + x]; count++; }
//     if (y < height - 1) { avg_u += P[(y + 1) * width + x]; count++; }

//     avg_u /= count;

//     // Aktualizacja przepływu
//     float alpha = 0.01f;
//     float flow = avg_u - (Ix * avg_u + Iy * avg_u + It) / (alpha + Ix * Ix + Iy * Iy + 0.0001f);

//     P[y * width + x] = flow;
// }

// // Makro do sprawdzania błędów CUDA
// #define CUDA_CHECK(call)                                                       \
//     {                                                                          \
//         cudaError_t err = call;                                                \
//         if (err != cudaSuccess)                                                \
//         {                                                                      \
//             fprintf(stderr, "CUDA Error: %s (code %d), line %d\n",             \
//                     cudaGetErrorString(err), err, __LINE__);                   \
//             exit(EXIT_FAILURE);                                                \
//         }                                                                      \
//     }

// void launch_optFlow_LucasKanade(float *h_N1, float *h_N2, float *h_P, const int height, const int width)
// {
//     float *d_N1 = nullptr, *d_N2 = nullptr, *d_P = nullptr;

//     // Alokacja pamięci na GPU
//     CUDA_CHECK(cudaMalloc(&d_N1, width * height * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_N2, width * height * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_P, width * height * sizeof(float)));

//     // Kopiowanie danych do GPU
//     CUDA_CHECK(cudaMemcpy(d_N1, h_N1, width * height * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_N2, h_N2, width * height * sizeof(float), cudaMemcpyHostToDevice));

//     // Definiowanie wymiarów bloków i siatki
//     dim3 blockSize(16, 16);
//     dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

//     // Uruchomienie jądra CUDA
//     optFlow_LucasKanade<<<gridSize, blockSize>>>(d_N1, d_N2, d_P, height, width);

    
//     // Sprawdzenie błędów po uruchomieniu jądra
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Kopiowanie wyników z GPU
//     CUDA_CHECK(cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost));


//     // Debugowanie pierwszych wyników po normalizacji
//     for (int i = 0; i < 10; i++) {
//         printf("Normalized P[%d] = %f\n", i, h_P[i]);
//     }

//     printf("Grid size: (%d, %d), Block size: (%d, %d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

//     // Zwolnienie pamięci na GPU
//     CUDA_CHECK(cudaFree(d_N1));
//     CUDA_CHECK(cudaFree(d_N2));
//     CUDA_CHECK(cudaFree(d_P));
// }


// void launch_optFlow_HornSchunck(float *h_N1, float *h_N2, float *h_P, const int height, const int width)
// {
//     float *d_N1 = nullptr, *d_N2 = nullptr, *d_P = nullptr;

//     // Alokacja pamięci na GPU
//     CUDA_CHECK(cudaMalloc(&d_N1, width * height * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_N2, width * height * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_P, width * height * sizeof(float)));

//     // Kopiowanie danych do GPU
//     CUDA_CHECK(cudaMemcpy(d_N1, h_N1, width * height * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_N2, h_N2, width * height * sizeof(float), cudaMemcpyHostToDevice));

//     // Definiowanie wymiarów bloków i siatki
//     dim3 blockSize(16, 16);
//     dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

//     // Uruchomienie jądra CUDA
//     optFlow_HornSchunck<<<gridSize, blockSize>>>(d_N1, d_N2, d_P, height, width);

//     // Sprawdzenie błędów po uruchomieniu jądra
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize());

//     // Kopiowanie wyników z GPU
//     CUDA_CHECK(cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost));
//     printf("Grid size: (%d, %d), Block size: (%d, %d)\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

//     // Zwolnienie pamięci
//     CUDA_CHECK(cudaFree(d_N1));
//     CUDA_CHECK(cudaFree(d_N2));
//     CUDA_CHECK(cudaFree(d_P));
// }


// int main(int argc, char *argv[])
// {
//     // Sprawdzenie liczby argumentów
//     if (argc != 3)
//     {
//         printf("Wrong number of arguments: 2 filenames are needed (input1.bmp and input2.bmp)\n");
//         return EXIT_FAILURE;
//     }

//     unsigned int width1, height1, width2, height2;

//     // Pobranie rozmiarów dla obu obrazów
//     getBMPSize(argv[1], &width1, &height1);
//     getBMPSize(argv[2], &width2, &height2);

//     // Sprawdzenie, czy rozmiary obu obrazów są takie same
//     if (width1 != width2 || height1 != height2)
//     {
//         fprintf(stderr, "Error: Input images must have the same dimensions.\n");
//         return EXIT_FAILURE;
//     }

//     unsigned int width = width1;
//     unsigned int height = height1;

//     // Alokacja pamięci dla obrazów
//     float *N1 = (float *)malloc(width * height * sizeof(float));
//     float *N2 = (float *)malloc(width * height * sizeof(float));
//     float *P = (float *)malloc(width * height * sizeof(float));

//     if (!N1 || !N2 || !P)
//     {
//         fprintf(stderr, "Error: Memory allocation failed.\n");
//         return EXIT_FAILURE;
//     }

//     // Wczytanie obrazów wejściowych (grayscale)
//     printf("Reading input images...\n");
//     readBMP(argv[1], N1, true);
//     readBMP(argv[2], N2, true);

//     printf("Running Lucas-Kanade Optical Flow...\n");
//     launch_optFlow_LucasKanade(N1, N2, P, height, width);
//     writeBMP("out_LukasKanade.bmp", P, width, height, true);
//     printf("Lucas-Kanade result saved to 'out_LukasKanade.bmp'\n");

//     printf("Running Horn-Schunck Optical Flow...\n");
//     launch_optFlow_HornSchunck(N1, N2, P, height, width);
//     writeBMP("out_HornSchunck.bmp", P, width, height, true);
//     printf("Horn-Schunck result saved to 'out_HornSchunck.bmp'\n");

//     // Zwolnienie pamięci
//     free(N1);
//     free(N2);
//     free(P);

//     printf("Program completed successfully.\n");
//     return EXIT_SUCCESS;
// }
