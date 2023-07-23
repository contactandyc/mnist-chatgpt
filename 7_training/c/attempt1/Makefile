CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -g
INCLUDES = -I.

SRCS = main.c image_utils.c data_processing.c
OBJS = $(SRCS:.c=.o)
TARGET = training

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)