# MEC

unofficial C++ implementation of "MEC: Memory-efficient Convolution for Deep Neural Network" published on ICML2017


## Requirement

install libatlas-base-dev:  `apt-get install libatlas-base-dev`

## Compiling

compile code and run

`g++ -o im2col im2col.cpp /usr/lib/x86_64-linux-gnu/atlas/libblas.so.3 -fopenmp`


## Reference

[1] [Im2Col+GEMM 的改进方法 MEC，一种更加高效的卷积计算策略](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96/Im2Col%2BGEMM%E7%9A%84%E6%94%B9%E8%BF%9B%E6%96%B9%E6%B3%95MEC%EF%BC%8C%E4%B8%80%E7%A7%8D%E6%9B%B4%E5%8A%A0%E9%AB%98%E6%95%88%E7%9A%84%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E7%AD%96%E7%95%A5/)
