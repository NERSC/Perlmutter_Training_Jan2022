NVCCFLAGS = -arch=sm_80
NVCFLAGS = -gpu=cc80

bcast_from_device: bcast_from_device.c
	cc -o $@ -cuda $<

clean:
	rm -f bcast_from_device
