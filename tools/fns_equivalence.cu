#include <cstdio>
#include <cstdint>
__device__ inline int ref_fnsb(unsigned a, int n, int& c32){
  int t, i=n, r=0; unsigned c1=a;
  unsigned c2=c1-((c1>>1)&0x55555555);
  unsigned c4=((c2>>2)&0x33333333)+(c2&0x33333333);
  unsigned c8=((c4>>4)+c4)&0x0f0f0f0f;
  unsigned c16=((c8>>8)+c8);
  c32=((c16>>16)+c16)&0x3f;
  t=(c16)&0x1f; if(i>=t){r+=16;i-=t;}
  t=(c8>>r)&0x0f; if(i>=t){r+=8;i-=t;}
  t=(c4>>r)&0x07; if(i>=t){r+=4;i-=t;}
  t=(c2>>r)&0x03; if(i>=t){r+=2;i-=t;}
  t=(c1>>r)&0x01; if(i>=t){r+=1;}
  if(n>=c32) r=-1;
  return r;
}
__device__ inline int fns_fnsb(unsigned a, int n, int& c32){
  c32 = __popc(a);
  return (int)__fns(a, 0u, n + 1);
}
__global__ void sweep(unsigned lo, unsigned hi, unsigned long long* bad, unsigned* firstbad){
  for(unsigned long long m = lo + (unsigned long long)blockIdx.x*blockDim.x + threadIdx.x;
      m < hi; m += (unsigned long long)gridDim.x*blockDim.x){
    unsigned mask = (unsigned)m;
    int pc = __popc(mask);
    for(int n = 0; n <= pc + 1; n++){       // include OUT OF RANGE on purpose
      int c1,c2;
      int a = ref_fnsb(mask,n,c1);
      int b = fns_fnsb(mask,n,c2);
      if(a != b || c1 != c2){
        if(atomicAdd(bad,1ULL)==0){ firstbad[0]=mask; firstbad[1]=(unsigned)n;
                                    firstbad[2]=(unsigned)a; firstbad[3]=(unsigned)b; }
      }
    }
  }
}
int main(){
  unsigned long long *dbad, hbad=0; unsigned *dfb, hfb[4]={0,0,0,0};
  cudaMalloc(&dbad,8); cudaMemset(dbad,0,8); cudaMalloc(&dfb,16); cudaMemset(dfb,0,16);
  // exhaustive over every 21-bit mask, every n including one past the end
  sweep<<<2048,256>>>(0u, 1u<<21, dbad, dfb);
  cudaDeviceSynchronize();
  cudaMemcpy(&hbad,dbad,8,cudaMemcpyDeviceToHost);
  printf("exhaustive masks 0..2^21-1, all n in [0,popc+1]: %llu mismatches\n", hbad);
  // and the top of the range, where the 32-bit edge cases live
  cudaMemset(dbad,0,8);
  sweep<<<2048,256>>>(0xFFE00000u, 0xFFFFFFFFu, dbad, dfb);
  cudaDeviceSynchronize();
  cudaMemcpy(&hbad,dbad,8,cudaMemcpyDeviceToHost);
  cudaMemcpy(hfb,dfb,16,cudaMemcpyDeviceToHost);
  printf("exhaustive masks 0xFFE00000..0xFFFFFFFE: %llu mismatches\n", hbad);
  if(hbad) printf("  first: mask=%08x n=%u ref=%d fns=%d\n",hfb[0],hfb[1],(int)hfb[2],(int)hfb[3]);
  return 0;
}
