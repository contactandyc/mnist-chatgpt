CC = gcc
CFLAGS = -Wall -Wextra -std=c99

all: one_hot_encoding

one_hot_encoding: one_hot_encoding.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f one_hot_encoding