#include "ppmIO.h"
#include "kernels.cu"

#include <stdio.h>
#include <stdlib.h>

void launch_optFlow_LucasKanade(float *h_N1, float *h_N2, float *h_P, const int height, const int width)
{
    //@@ INSERT CODE HERE
}

void launch_optFlow_HornSchunck(float *h_N1, float *h_N2, float *h_P, const int height, const int width)
{
    //@@ INSERT CODE HERE
}

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input filename + mask size
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 2 arguments needed (input filename one and input filename two)\n");
        return 1;
    }

    char outLKName[] = "out_LukasKanade.ppm";
    char outHSName[] = "out_HornSchunck.ppm";

    //@@ INSERT CODE HERE

    return 0;
}