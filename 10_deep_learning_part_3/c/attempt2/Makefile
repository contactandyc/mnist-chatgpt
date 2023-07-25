CC = gcc
# CFLAGS = -g -fsanitize=address
CFLAGS =  -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL
INCLUDES = -I.

SRCS = main.c activation.c matrix.c image_utils.c data_processing.c neural_network.c ac_allocator.c
OBJS = $(SRCS:.c=.o)
TARGET = train_and_test

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)