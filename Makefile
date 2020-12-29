# Name
NAME = barnes_hut
 
# Compiler
CXX = mpicc
EXT = c
FLAGS = -lm
 
all:
	$(CXX) $(NAME).$(EXT) -o $(NAME) $(FLAGS)
 
clean:
	rm -rf *.o $(NAME)