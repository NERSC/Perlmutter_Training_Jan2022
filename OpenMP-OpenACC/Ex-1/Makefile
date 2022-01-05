SHELL = /bin/sh

#==========================
# Files
#==========================
SRC = *.cpp

CXX = nvc++

OPENMP:=n
OPENACC:=n

CXXFLAGS += -O3 

ifeq ($(OPENMP),y)
CXXFLAGS += -mp=gpu -gpu=cc80 -Minfo
EXE = vec_add.openmp
else ifeq ($(OPENACC),y)
CXXFLAGS += -acc -Minfo=accel
EXE = vec_add.openacc
else
EXE = vec_add.seq
endif


#==========================
# Compiler commands
#==========================
CXXLD         = $(CXX)

#==========================
# Make the executable
#==========================
$(EXE): $(SRC)
	echo $(SRC)
	$(CXXLD) $(CXXFLAGS) $(SRC) -o $(EXE)

#==========================
#remove all objs
#==========================
clean:
	/bin/rm -f *.o vec_add.seq vec_add.openmp vec_add.openacc
