CC = gcc
CFLAGS = -Wall -Wextra -std=c99

TARGET = convert_pixels_to_floats

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
