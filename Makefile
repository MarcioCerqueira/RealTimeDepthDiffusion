.SUFFIXES: .c .cc .cpp .cxx .h

# compiler names:
CXX		= g++ 
CC		= gcc

# flags for C++ compiler:
CFLAGS		= -g 
CXXFLAGS	= -g -std=c++11

# libraries to link with:

INC_PATH			=	-I/usr/include/eigen3 -I/usr/include/trilinos \
						-I/usr/lib/x86_64-linux-gnu/openmpi/include
LIB_PATH 			=	-L/usr/lib/nvidia-375/ \
						-L/usr/lib/x86_64-linux-gnu/ \
						-L/usr/local/lib 

GL_LDFLAGS 			=	-lGL -lGLU -lGLEW
GLFW_LDFLAGS 		=	-lglfw
GLUT_LDFLAGS 		=	-lglut
ASSIMP_LDFLAGS 		=	-lassimp
SOIL_LDFLAGS		= 	-lSOIL
OPENMP_LDFLAGS		= 	-fopenmp
OPENCV_LDFLAGS		= 	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
TRILINOS_LDFLAGS	= 	-lmpi -lmpi_cxx -ltrilinos_epetra -ltrilinos_epetraext \
						-ltrilinos_teuchoscore -ltrilinos_teuchoscore \
						-ltrilinos_teuchoscomm -ltrilinos_teuchosparameterlist \
						-ltrilinos_ml -ltrilinos_aztecoo  

OBJFILES 			=	main.o Solver.o LAHBPCG.o

# compile rules
.c.o:	$*.h
	@echo "Compiling C code ...."
	$(CC) -o $*.o -c $(CXXFLAGS) $(INC_PATH) $*.c

.cpp.o:	$*.h
	@echo "Compiling C++ code ...."
	$(CXX) -o $*.o -c $(CXXFLAGS) $(INC_PATH) $*.cpp

# ***********************************************************************************
all:	main

main:  $(OBJFILES)
	@echo "Linking ...."
	$(CXX) $^ -o $@ $(LIB_PATH) $(OPENMP_LDFLAGS) $(OPENCV_LDFLAGS) $(TRILINOS_LDFLAGS)
		
clean:	
	@echo "Clearing ..."
	rm -f *.o core *.*~ *~ main
