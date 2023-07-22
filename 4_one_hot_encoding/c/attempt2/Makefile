CC = gcc
CFLAGS = -Wall -Wextra -std=c99

all: main

main: main.c one_hot_encoding.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f main
