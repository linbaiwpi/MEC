#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <fstream>
#include <string>
#include <omp.h>
#define THREAD_NUM 1

extern "C" {
    #include <cblas.h>
}
using namespace std;

void im2col_mec(float** src,
                const int &inH, const int &inW,
                const int &kH, const int &kW,
                float* dest, bool col_raster = false) {
    const int outH = inH - kH + 1;
    const int outW = inW - kW + 1;

    int idx = 0;
    if (col_raster) {
        // row raster
        for (int j = 0; j < outW; ++j) {
            for (int i = 0; i < inH; ++i) {
                for(int k = j; k < j + kW; k++) {
                    dest[idx++] = src[i][k];
                }
            }
        }
        cout << "IM2COL ===================" << endl;
        for (int i = 0; i < outH; i++) {
            for (int j = 0; j < inH * kW; j++) {
                cout << dest[i*inH*kW+j] << ",";
            }
            cout << endl;
        }
    } else {
        // col raster
        idx = 0;
        for (int j = 0; j < outW; ++j) {
            for (int i = 0; i < inH; ++i) {
                for(int k = j; k < j + kW; k++) {
                    idx = ((k - j) + i * kW) * outW + j;
                    dest[idx] = src[i][k];
                }
            }
        }
        cout << "IM2COL ===================" << endl;
        for (int i = 0; i < inH * kW; i++) {
            for (int j = 0; j < outH; j++) {
                cout << dest[i*outH+j] << ",";
            }
            cout << endl;
        }
    }
}


const int kernel_num = 1;
const int kernel_h = 5;
const int kernel_w = 5;
const int inHeight = 10;
const int inWidth = 10;

void printMemoryUsage() {
    std::ifstream procFile("/proc/self/status");
    std::string line;
    while (std::getline(procFile, line)) {
        if (line.find("VmRSS:") == 0) {  // 查找 VmRSS 行
            std::cout << line << std::endl;
            break;
        }
    }
}

int main(){
    // 构造输入矩阵
    float **src = new float*[inHeight];
    for(int i = 0; i < inHeight; i++){
        src[i] = new float[inWidth];
        for(int j = 0; j < inWidth; j++){
            src[i][j] = (i * inWidth + j) * 0.1;
        }
    }
    cout << "SOURCE ===================" << endl;
    for (int i = 0; i < inHeight; i++) {
        for (int j = 0; j < inWidth; j++) {
            cout << src[i][j] << ",";
        }
        cout << endl;
    }

    // 构造kernel矩阵
    float **kernel[kernel_num];
    for(int i = 0; i < kernel_num; i++){
        kernel[i] = new float*[kernel_h];
        for(int j = 0; j < kernel_h; j++){
            kernel[i][j] = new float[kernel_w];
            for(int k = 0; k < kernel_w; k++){
                kernel[i][j][k] = 0.2;
            }
        }
    }

    // 开始计时
    struct timeval tstart, tend;
    gettimeofday(&tstart, NULL);

    // 对kernel进行Im2col
    float* kernel2col = new float[kernel_num*kernel_h*kernel_w];
    int cnt = 0;
    for(int i = 0; i < kernel_num; i++){
        for(int j = 0; j < kernel_h; j++){
            for(int k = 0; k < kernel_w; k++){
                kernel2col[cnt++] = kernel[i][j][k];
            }
        }
    }
    cout << "kernel size = " << float(kernel_num*kernel_h*kernel_w * 4) / 1000000 << "MB" << endl;
    // 对输入矩阵Im2col
    int outHeight = inHeight - kernel_h + 1;
    int outWidth = inWidth - kernel_w + 1;
    float *srcIm2col = new float[outWidth * inHeight * kernel_w];
    cout << "im2col size = " << float(kernel_w * kernel_h * outWidth * outHeight * 4) / 1000000 << "MB" << endl;
    im2col_mec(src, inHeight, inWidth, kernel_h, kernel_w, srcIm2col);

    // 使用Blas库实现矩阵乘法
    float **output = new float*[outHeight];

#pragma omp parallel for num_threads(THREAD_NUM)
    for(int i = 0; i < outHeight; i++){
        output[i] = new float [kernel_num * outWidth];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,kernel_num,
            outWidth, kernel_w * kernel_h,1,
            kernel2col, kernel_h * kernel_w,
            srcIm2col + i * outWidth, outWidth, 0, output[i], outWidth);
    }
    printMemoryUsage();

    cout << "OUTPUT ===================" << endl;
    for (int i = 0; i < outHeight; i++) {
        for (int j = 0; j < outWidth; j++) {
            cout << output[i][j] << ",";
        }
    }
    cout << endl;

    // 结束计时
    gettimeofday(&tend, NULL);
    cout<<"MEC Total time cost: "<<(tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_usec-tstart.tv_usec)/1000<<" ms"<<endl;

    // 
    delete [] kernel2col;
    delete [] srcIm2col;
    
    for(int i = 0; i < outHeight; i++){
        delete [] output[i];
    }

    for(int i = 0; i < kernel_num; i++){
        for(int j = 0; j < kernel_h; j++){
            delete [] kernel[i][j];
        }
        delete [] kernel[i];
    }

    for(int i = 0; i < inHeight; i++){
        delete [] src[i];
    }

    return 0;
}