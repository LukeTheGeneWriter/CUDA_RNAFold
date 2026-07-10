//WBL 20 Jan 2018

//https://devtalk.nvidia.com/default/topic/799429/cuda-programming-and-performance/possible-to-use-the-cuda-math-api-integer-intrinsics-to-find-the-nth-unset-bit-in-a-32-bit-int/1
//njuffa Posted 12/29/2014 08:48 PM
//    int find_nth_clear_bit (unsigned int a, int n)
//WBL removed 2nd line for find_nth_set_bit and c32 argument
__device__ inline
int find_nth_set_bit (unsigned int a, int n, int& c32 /*popc*/)
    {
        int t, i = n, r = 0;
        unsigned int c1 = a;
//      unsigned int c1 = ~a; // search for 1-bits instead of 0-bits
        unsigned int c2 = c1 - ((c1 >> 1) & 0x55555555);
        unsigned int c4 = ((c2 >> 2) & 0x33333333) + (c2 & 0x33333333);
        unsigned int c8 = ((c4 >> 4) + c4) & 0x0f0f0f0f;
        unsigned int c16 = ((c8 >> 8) + c8);
        /*int*/ c32 = ((c16 >> 16) + c16) & 0x3f;
        t = (c16    ) & 0x1f; if (i >= t) { r += 16; i -= t; }
        t = (c8 >> r) & 0x0f; if (i >= t) { r +=  8; i -= t; }
        t = (c4 >> r) & 0x07; if (i >= t) { r +=  4; i -= t; }
        t = (c2 >> r) & 0x03; if (i >= t) { r +=  2; i -= t; }
        t = (c1 >> r) & 0x01; if (i >= t) { r +=  1;         }
        if (n >= c32) r = -1;
        return r; 
    }
