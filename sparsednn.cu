%%cu
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#define row 60000
#define column 1024

using namespace std;
using namespace boost::algorithm;

__global__ void ReluActivation(float * MNIST,float* layer,float * Out)
{
    
    int Row=blockIdx.y*blockDim.y+threadIdx.y;
    int Column=blockIdx.x*blockDim.x+threadIdx.x;
    if(Row<60000 && Column < 1024)
    {
        float bias=0;
        for(int z=0;z<1024;z++)   bias+=MNIST[Row*1024+z]*layer[z*1024+Column];
        bias = bias - 0.3;
     
        if((bias)<0)  bias = 0;
        if((bias) >32)  bias = 32;
        Out[Row*1024+Column]=bias;
    }
    __syncthreads();
}
int main() {

    ifstream file("drive/MyDrive/sparse-images-1024.tsv");
    string line;
    int counter = 0;
    float *Y_zero;
    Y_zero  = (float*)malloc(60000*1024*sizeof(float));
    for(int i=0; i<row*column; i++)    Y_zero[i]=0;
    
    while (getline(file, line)) {
        counter++;
        vector<string> parts;
        split(parts, line, boost::is_any_of("\t"));
        string st1 = parts[0];
        int r = stoi(st1);
        string st2 = parts[1];
        int c = stoi(st2);

        Y_zero[(r-1)*1024+(c-1)]=1;
    }
    file.close();

    printf("TNZC: %d ", counter);

    for(int i=1; i<=120; i++){
      string  nlayer="n1024-l";
      string lay=to_string(i);
      nlayer="drive/MyDrive/neuron1024/"+nlayer+lay+".tsv";
      ifstream files(nlayer);
      string line;
      float *weight;
      weight=(float*)malloc(1024*1024*sizeof(float));
      for(int i=0; i<1024*1024; i++)     weight[i]=0;


    while (getline(files, line)) {
        vector<string> parts;
        split(parts, line, boost::is_any_of("\t"));
        // TODO Your code goes here!
        string st1 = parts[0];
        int ro = stoi(st1);
        string st2 = parts[1];
         int co = stoi(st2);
        weight[(ro-1)*1024+(co-1)]=0.0625;
    }
  
    files.close();

    float *D_MNIST;
    cudaMalloc(&D_MNIST,60000*1024*sizeof(float));
    float *D_Weight;
    cudaMalloc(&D_Weight,1024*1024*sizeof(float));
    float* D_out;
    cudaMalloc(&D_out,60000*1024*sizeof(float)); 
    cudaMemcpy(D_MNIST,Y_zero,60000*1024*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(D_Weight,weight,1024*1024*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimGrid(32, 1875);
    dim3 dimBlock(32, 32);
    /* Start measuring time
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    */

    ReluActivation<<<dimGrid, dimBlock>>>(D_MNIST, D_Weight, D_out);
    cudaDeviceSynchronize(); 

   /* Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;
    
    printf("Time measured: %.6f seconds.\n", elapsed);


*/
      cudaMemcpy(Y_zero,D_out,60000*1024*sizeof(float),cudaMemcpyDeviceToHost);
          int g=0;
    for(int i=0;i<1024;i++){
        for(int j=0;j<60000;j++){
            if(Y_zero[i*60000+j]!=0){
                g++;        //calculating the non- zero elements
            }
        }
    }
    printf("%d ",g);
}
int active_images=0;
  for(int i=0;i<60000;i++)
  {
      for(int j=0;j<1024;j++)
      {
          if(Y_zero[i*1024+j]!=0)
          {
          printf("%d\n ",i+1);
           active_images++;
            break;
          }
      }
  }
    printf("\nTotal Active images %d           ",active_images);
    return 0;
}