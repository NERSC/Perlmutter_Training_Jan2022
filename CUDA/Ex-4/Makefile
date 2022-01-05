NVCCFLAGS = -arch=sm_80
NVCFLAGS = -gpu=cc80

vec_add: kernels.o kernels.h vecAdd.cpp
	CC -o $@ -L$(CUDA_HOME)/lib64 -L/opt/cray/pe/mpich/8.1.11/gtl/lib -lmpi_gtl_cuda -lcudart vecAdd.cpp kernels.o
kernels.o: kernels.cu kernels.h
	nvcc $(NVCCFLAGS) -c kernels.cu -o $@
clean:
	rm *.o vec_add
