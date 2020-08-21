COMMON=-O2 -I../include -L../bin -std=c++11 -mavx

default:
	g++ $(COMMON) test.cpp -lmujoco150nogl -o ../bin/test
	g++ $(COMMON) compile.cpp -lmujoco150nogl -o ../bin/compile
	g++ $(COMMON) derivative.cpp -lmujoco150nogl -fopenmp -o ../bin/derivative
	g++ $(COMMON) simulate.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/simulate
	g++ $(COMMON) record.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/record
	g++ $(COMMON) basic.cpp -lmujoco150 -lGL -lglew ../bin/libglfw.so.3 -o ../bin/basic

egl:	
	g++ $(COMMON) -DMJ_EGL record.cpp -lmujoco150 -lOpenGL -lEGL -lglewegl -o ../bin/recordegl

osmesa:	
	g++ $(COMMON) -DMJ_OSMESA record.cpp -lmujoco150 -lOSMesa -lglewosmesa -o ../bin/recordosmesa

all: default egl osmesa
