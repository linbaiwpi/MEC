#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <fstream>
#include <string>

extern "C" {
    #include <cblas.h>
}
using namespace std;

void im2col_cpu(float** src,
                const int &inH, const int &inW,
                const int &kH, const int &kW,
                float* dest, bool col_raster = true) {
    const int outH = inH - kH + 1;
    const int outW = inW - kW + 1;

    int stH = 1, stW = 1;
    int dH = 0, dW = 0; // not supported yet
    int pH = 0, pW = 0;

    int idx = 0;
    if (col_raster) {
        // column raster
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                for (int p = 0; p < outH; ++p) {
                    int idxH = p * stH - pH;
                    for (int q = 0; q < outW; ++q) {
                        int idxW = q * stW - pW;
                        if ((idxH + i < 0) || (idxH + i > inH - 1) ||
                            (idxW + j < 0) || (idxW + i > inW - 1)) {
                            dest[idx++] = 0;
                        }
                        else {
                            dest[idx++] = src[idxH + i][idxW + j];
                        }
                        // cout << "dest[" << idx-1 << "] = src[" << idxH + i << "][" << idxW + j << "]" << endl;
                    }
                }
            }
        }
        // cout << "IM2COL ===================" << endl;
        // for (int i = 0; i < kH*kW; i++) {
        //     for (int j = 0; j < outH*outW; j++) {
        //         cout << dest[i*outH*outW+j] << ",";
        //     }
        //     cout << endl;
        // }
    } else {
        // row raster
        for (int p = 0; p < outH; ++p) {
            int idxH = p * stH - pH;
            for (int q = 0; q < outW; ++q) {
                int idxW = q * stW - pW;
                // idx = idxH * outW + idxW;
                for (int i = 0; i < kH; ++i) {
                    for (int j = 0; j < kW; ++j) {
                        if ((idxH + i < 0) || (idxH + i > inH - 1) ||
                            (idxW + j < 0) || (idxW + i > inW - 1)) {
                            dest[idx++] = 0;
                            // cout << "= dest[" << idx-1 << "] = src[" << idxH + i << "][" << idxW + j << "]" << endl;
                        }
                        else {
                            dest[idx++] = src[idxH + i][idxW + j];
                            // cout << "dest[" << idx-1 << "] = src[" << idxH + i << "][" << idxW + j << "]" << endl;
                        }
                    }
                }
            }
        }
        // cout << "IM2COL ===================" << endl;
        // for (int i = 0; i < outH*outW; i++) {
        //     for (int j = 0; j < kH*kW; j++) {
        //         cout << dest[i*kH*kW+j] << ",";
        //     }
        //     cout << endl;
        // }
    }
}

const int kernel_num = 64;
const int kernel_h = 7;
const int kernel_w = 7;
const int inHeight = 224;
const int inWidth = 224;

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
    // construction input matrix
    float **src = new float*[inHeight];
    for(int i = 0; i < inHeight; i++){
        src[i] = new float[inWidth];
        for(int j = 0; j < inWidth; j++){
            src[i][j] = (i * inWidth + j) * 0.1;
        }
    }
    // cout << "SOURCE ===================" << endl;
    // for (int i = 0; i < inHeight; i++) {
    //     for (int j = 0; j < inWidth; j++) {
    //         cout << src[i][j] << ",";
    //     }
    //     cout << endl;
    // }

    // construct weight matrix
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
    float *srcIm2col = new float[kernel_w * kernel_h * outWidth * outHeight];
    cout << "im2col size = " << float(kernel_w * kernel_h * outWidth * outHeight * 4) / 1000000 << "MB" << endl;
    im2col_cpu(src, inHeight, inWidth, kernel_h, kernel_w, srcIm2col);

    // 使用Blas库实现矩阵乘法
    float *output = new float[kernel_num * outHeight * outWidth];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    kernel_num, outHeight*outWidth, kernel_w*kernel_h, 
    1, kernel2col, kernel_h*kernel_w, srcIm2col,outHeight * outWidth, 0,
    output, outHeight * outWidth);
    printMemoryUsage();

    // cout << "OUTPUT ===================" << endl;
    // for (int i = 0; i < outHeight; i++) {
    //     for (int j = 0; j < outWidth; j++) {
    //         cout << output[i*outWidth+j] << ",";
    //     }
    // }
    // cout << endl;    

    // 结束计时
    gettimeofday(&tend, NULL);
    cout<<"im2colOrigin Total time cost: "<<(tend.tv_sec-tstart.tv_sec)*1000 + (tend.tv_usec-tstart.tv_usec)/1000<<" ms"<<endl;

    // 
    delete [] kernel2col;
    delete [] srcIm2col;
    delete [] output;

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