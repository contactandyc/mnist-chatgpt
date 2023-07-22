CC = gcc
CFLAGS = -Wall -Wextra -std=c99

SRCS = main.c
OBJS = $(SRCS:.c=.o)
TARGET = file_reader

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
