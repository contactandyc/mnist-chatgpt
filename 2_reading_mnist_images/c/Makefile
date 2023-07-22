CC = gcc
CFLAGS = -Wall -Wextra -std=c99

TARGET = file_reader

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
