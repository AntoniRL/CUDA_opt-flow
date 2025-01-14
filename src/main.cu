#include "bmpIO.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <string>

/**
 * Kernel do obliczania gradientów Ix, Iy oraz różnicy It = N2 - N1
 * zakładamy, że:
 *   - dN1, dN2 to obrazy wejściowe w GPU
 *   - dIx, dIy, dIt to tablice, do których zapisujemy wyniki
 *   - width, height rozmiary obrazka
 *
 * Używamy prostego operatora różnicowego:
 *   Ix = (I(x+1,y) - I(x-1,y)) / 2
 *   Iy = (I(x,y+1) - I(x,y-1)) / 2
 */
__global__ void computeGradients(const float* __restrict__ dN1,
                                 const float* __restrict__ dN2,
                                 float* __restrict__ dIx,
                                 float* __restrict__ dIy,
                                 float* __restrict__ dIt,
                                 int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1) return;
    if (y < 1 || y >= height - 1) return;

    int idx = y * width + x;

    float Ix = (dN1[y * width + (x+1)] - dN1[y * width + (x-1)]) * 0.5f;
    float Iy = (dN1[(y+1) * width + x] - dN1[(y-1) * width + x]) * 0.5f;
    float It = dN2[idx] - dN1[idx];

    dIx[idx] = Ix;
    dIy[idx] = Iy;
    dIt[idx] = It;
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
__global__ void lucasKanadeKernel(const float* __restrict__ dIx,
                                  const float* __restrict__ dIy,
                                  const float* __restrict__ dIt,
                                  float* __restrict__ dU,
                                  float* __restrict__ dV,
                                  int width, int height,
                                  int r) // promień okienka
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Pomijamy piksele na brzegu
    if (x < r || x >= width - r) return;
    if (y < r || y >= height - r) return;

    float A11 = 0.0f, A12 = 0.0f, A21 = 0.0f, A22 = 0.0f;
    float B1  = 0.0f, B2  = 0.0f;

    // Sumowanie w małym oknie
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int xx = x + dx;
            int yy = y + dy;
            int idx = yy * width + xx;

            float ix = dIx[idx];
            float iy = dIy[idx];
            float it = dIt[idx];

            // Macierz A = sum( [ix^2  ix*iy ; ix*iy  iy^2 ] )
            A11 += ix * ix;
            A12 += ix * iy;
            A21 += ix * iy;
            A22 += iy * iy;

            // Wektor B = - sum( [ix*it ; iy*it] )
            B1  -= ix * it;
            B2  -= iy * it;
        }
    }

    // Wyznacznik
    float det = A11 * A22 - A12 * A21;
    if (fabs(det) < 1e-6f) {
        // Jeśli macierz prawie osobliwa, u = v = 0
        dU[y * width + x] = 0.0f;
        dV[y * width + x] = 0.0f;
        return;
    }

    // Rozwiązanie układu 2x2
    float u = (A22 * B1 - A12 * B2) / det;
    float v = (A11 * B2 - A21 * B1) / det;

    dU[y * width + x] = u;
    dV[y * width + x] = v;
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
    std::string method = argv[3];

    // --- Wczytanie bitmap
    unsigned int width, height;
    getBMPSize(file1, &width, &height);

    float* hN1 = (float*)malloc(width * height * sizeof(float));
    float* hN2 = (float*)malloc(width * height * sizeof(float));

    readBMP(file1, hN1, /*isGray=*/true);
    readBMP(file2, hN2, /*isGray=*/true);


    printf("Wczytano plik1: %s\n", file1);
    printf("Wczytano plik2: %s\n", file2);
    printf("width = %d, height = %d\n", width, height);

    // Wypisz kilka pierwszych pikseli obrazu 1
    printf("Pierwsze 10 pikseli obrazu 1 (hN1):\n");
    for (int i = 0; i < 10 && i < width*height; i++) {
        printf("%.1f ", hN1[i]);
    }
    printf("\n");

    // Wypisz kilka pierwszych pikseli obrazu 2 (hN2):
    printf("Pierwsze 10 pikseli obrazu 2 (hN2):\n");
    for (int i = 0; i < 10 && i < width*height; i++) {
        printf("%.1f ", hN2[i]);
    }
    printf("\n");

    // Opcjonalnie sprawdź różnicę piksel po pikselu
    printf("Pierwsze 10 różnic (N2 - N1):\n");
    for (int i = 0; i < 10 && i < width*height; i++) {
        float diff = hN2[i] - hN1[i];
        printf("%.1f ", diff);
    }
    printf("\n");


    // --- Alokacja GPU
    float *dN1, *dN2, *dIx, *dIy, *dIt;
    cudaMalloc((void**)&dN1,  width * height * sizeof(float));
    cudaMalloc((void**)&dN2,  width * height * sizeof(float));
    cudaMalloc((void**)&dIx,  width * height * sizeof(float));
    cudaMalloc((void**)&dIy,  width * height * sizeof(float));
    cudaMalloc((void**)&dIt,  width * height * sizeof(float));

    cudaMemcpy(dN1, hN1, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dN2, hN2, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Obliczenie gradientów
    dim3 block(16, 16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    computeGradients<<<grid, block>>>(dN1, dN2, dIx, dIy, dIt, width, height);
    cudaDeviceSynchronize();

        // Skopiuj gradienty na CPU
    float* hIx = (float*)malloc(width * height * sizeof(float));
    float* hIy = (float*)malloc(width * height * sizeof(float));
    float* hIt = (float*)malloc(width * height * sizeof(float));

    cudaMemcpy(hIx, dIx, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hIy, dIy, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hIt, dIt, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("DEBUG: Po computeGradients, Ix od indeksu 257 do 266:\n");
    int start = width + 1;  // (x=1, y=1) w pamięci, bo y*width + x
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", hIx[start + i]);
    }
    printf("\n");



    // Przygotowanie tablic na wyniki (U, V)
    float *dU, *dV; 
    cudaMalloc((void**)&dU, width * height * sizeof(float));
    cudaMalloc((void**)&dV, width * height * sizeof(float));
    cudaMemset(dU, 0, width * height * sizeof(float));
    cudaMemset(dV, 0, width * height * sizeof(float));

    // Hostowe tablice na wynik przepływu
    float* hU = (float*)malloc(width * height * sizeof(float));
    float* hV = (float*)malloc(width * height * sizeof(float));

    if (method == "LK") {
        printf("=== Lucas–Kanade ===\n");
        // Ustawmy promień okna (np. 2 => okno 5x5)
        int r = 10;
        lucasKanadeKernel<<<grid, block>>>(dIx, dIy, dIt, dU, dV, width, height, r);
        cudaDeviceSynchronize();

        // Zczytujemy U, V
        cudaMemcpy(hU, dU, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hV, dV, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        printf("DEBUG: Po %s, pierwsze 100 pikseli (U, V):\n", method.c_str());
        for (int i = 0; i < 100; i++) {
            printf("(%7.3f, %7.3f) ", hU[i], hV[i]);
        }
        printf("\n");
        int nonZeroCount = 0;

        for (int i = 0; i < width*height; i++) {
            if (fabs(hU[i]) > 1e-5f || fabs(hV[i]) > 1e-5f) nonZeroCount++;
        }
        printf("Liczba pikseli, gdzie (u,v) != 0: %d\n", nonZeroCount);

    } else if (method == "HS") {
        printf("=== Horn–Schunck ===\n");
        // Alokacja tymczasowych tablic do iteracyjnej aktualizacji
        float *dUnew, *dVnew;
        cudaMalloc((void**)&dUnew, width * height * sizeof(float));
        cudaMalloc((void**)&dVnew, width * height * sizeof(float));

        float alpha = 100.0f;    // regularization
        int   nIter = 1000;     // liczba iteracji

        for (int i = 0; i < nIter; i++) {
            hornSchunckIteration<<<grid, block>>>(
                dIx, dIy, dIt,
                dU, dV,         // stara
                dUnew, dVnew,   // nowa
                width, height, alpha
            );
            cudaDeviceSynchronize();

            // Zamiana wskaźników (dUnew -> dU, dVnew -> dV)
            // zamiast kopiować pamięć, robimy "swap"
            float* tmpU = dU; dU = dUnew; dUnew = tmpU;
            float* tmpV = dV; dV = dVnew; dVnew = tmpV;
        }

        // Po iteracjach w dU, dV mamy wynik
        cudaMemcpy(hU, dU, width * height * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hV, dV, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        printf("DEBUG: Po %s, pierwsze 100 pikseli (U, V):\n", method.c_str());
        for (int i = 0; i < 100; i++) {
            printf("(%7.3f, %7.3f) ", hU[i], hV[i]);
        }
        printf("\n");

        cudaFree(dUnew);
        cudaFree(dVnew);

    } else {
        printf("Metoda nie rozpoznana! Użyj 'LK' lub 'HS'.\n");
        return 0;
    }

    // Tworzymy wynikowy obraz – magnituda przepływu
    float* hFlow = (float*)malloc(width * height * sizeof(float));
    for (unsigned int i = 0; i < width * height; i++) {
        float mag = sqrtf(hU[i]*hU[i] + hV[i]*hV[i]);
        hFlow[i] = mag;
    }

    #include <float.h> // dla FLT_MAX itd.

    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;

    for (int i = 0; i < width * height; i++) {
        if (hFlow[i] < minVal) minVal = hFlow[i];
        if (hFlow[i] > maxVal) maxVal = hFlow[i];
    }
    printf("Zakres flow: min=%.6f, max=%.6f\n", minVal, maxVal);

    int countOver = 0;
    float threshold = 0.5f;
    for (int i = 0; i < width * height; i++) {
        if (hFlow[i] > threshold) {
            countOver++;
        }
    }
    printf("Liczba pikseli, gdzie flow > %.1f = %d\n", threshold, countOver);
    float scale = 255.0f / 0.818741f;  // np. dynamicznie liczone: (255.0f / maxVal)
    for (int i = 0; i < width*height; i++) {
        hFlow[i] *= scale;
    }
    // teraz w hFlow mamy zakres ~0..255

    // Zapis do BMP
    writeBMP("flow_magnitude.bmp", hFlow, width, height, true);
    printf("DEBUG: Magnituda przepływu, pierwsze 10:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", hFlow[i]);
    }
    printf("\n");


    // Opcjonalnie odczytajmy z powrotem i wyświetlmy parę pikseli:
    float* hFlowCheck = (float*)malloc(width * height * sizeof(float));
    readBMP("flow_magnitude.bmp", hFlowCheck, true);

    printf("Pierwsze 10 pikseli obrazu zapisanego (flow_magnitude.bmp):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", hFlowCheck[i]);
    }
    printf("\n");

    free(hFlowCheck);
    printf("Wynik zapisano do flow_magnitude.bmp\n");

    // Zwolnienie pamięci
    free(hN1);
    free(hN2);
    free(hU);
    free(hV);
    free(hFlow);

    cudaFree(dN1);
    cudaFree(dN2);
    cudaFree(dIx);
    cudaFree(dIy);
    cudaFree(dIt);
    cudaFree(dU);
    cudaFree(dV);

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
