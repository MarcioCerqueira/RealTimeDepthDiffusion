.SUFFIXES: .c .cc .cpp .cxx .h

# compiler names:
CXX		= g++ 
CC		= gcc

# flags for C++ compiler:
CFLAGS		= -g 
CXXFLAGS	= -g -std=c++11

# libraries to link with:

INC_PATH		=	-I/usr/include/eigen3
LIB_PATH 		=	-L/usr/lib/nvidia-375/ \
					-L/usr/lib/x86_64-linux-gnu/ \
					-L/usr/local/lib

GL_LDFLAGS 		=	-lGL -lGLU -lGLEW
GLFW_LDFLAGS 	=	-lglfw
GLUT_LDFLAGS 	=	-lglut
ASSIMP_LDFLAGS 	=	-lassimp
SOIL_LDFLAGS	= 	-lSOIL
OPENMP_LDFLAGS	= 	-fopenmp
OPENCV_LDFLAGS	= 	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

OBJFILES 		=	main.o Solver.o

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
	$(CXX) $^ -o $@ $(LIB_PATH) $(OPENMP_LDFLAGS) $(OPENCV_LDFLAGS)
		
clean:	
	@echo "Clearing ..."
	rm -f *.o core *.*~ *~ main
