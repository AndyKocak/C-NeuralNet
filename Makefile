# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#
# for C++ define  CC = g++
CC = gcc
CFLAGS  = -g -Wall

# typing 'make' will invoke the first target entry in the file 
# (in this case the default target entry)
# you can name this target entry anything, but "default" or "all"
# are the most commonly used names by convention
#
default: test

test:	example1.o neural.o helpers.o
	$(CC) $(CFLAGS) -o test example1.o neural.o helpers.o

example1.o:	example1.c neural.h helpers.h
	$(CC) $(CFLAGS) -c example1.c

neural.o:	neural.c neural.h helpers.h
	$(CC) $(CFLAGS) -c neural.c

helpers.o:	helpers.c helpers.h
	$(CC) $(CFLAGS) -c helpers.c

# To start over from scratch, type 'make clean'.  This
# removes the executable file, as well as old .o object
# files and *~ backup files:
#
clean: 
	$(RM) count *.o *~