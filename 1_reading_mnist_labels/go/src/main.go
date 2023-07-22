package main

import (
	"fmt"
	"log"
	"os"
)

// FileResponse represents the data structure to hold the contents of the file.
type FileResponse struct {
	MagicNumber int    // The magic number read from the file header.
	NumItems    int    // The number of items read from the file header.
	Labels      []byte // An array to store the label values.
}

// readFile reads the file with the given fileName and returns a pointer to FileResponse.
func readFile(fileName string) (*FileResponse, error) {
	file, err := os.Open(fileName) // Open the file in read mode.
	if err != nil {
		return nil, err // Return error if file cannot be opened.
	}
	defer file.Close() // Defer closing the file until the function returns.

	response := &FileResponse{}
	headerBytes := make([]byte, 8) // A buffer to read the first 8 bytes (magic number and num_items).

	// Read the first 8 bytes (magic number and num_items) from the file header.
	_, err = file.Read(headerBytes)
	if err != nil {
		return nil, err // Return error if reading the header fails.
	}

	// Convert the first 4 bytes to the magic number (big-endian encoding).
	response.MagicNumber = int(headerBytes[0])<<24 | int(headerBytes[1])<<16 | int(headerBytes[2])<<8 | int(headerBytes[3])

	// Convert the next 4 bytes to the number of items (big-endian encoding).
	response.NumItems = int(headerBytes[4])<<24 | int(headerBytes[5])<<16 | int(headerBytes[6])<<8 | int(headerBytes[7])

	// Read the label bytes from the file.
	response.Labels = make([]byte, response.NumItems)
	_, err = file.Read(response.Labels)
	if err != nil {
		return nil, err // Return error if reading the labels fails.
	}

	return response, nil // Return the FileResponse pointer with data.
}

// processFile reads the file with the given fileName and prints its contents.
func processFile(fileName string) {
	response, err := readFile(fileName) // Read the file contents.
	if err != nil {
		log.Fatalf("Error reading file %s: %v", fileName, err) // Handle error if reading fails.
	}

	fmt.Printf("File: %s\n", fileName)
	fmt.Printf("Magic Number: %d\n", response.MagicNumber)
	fmt.Printf("Number of Items: %d\n", response.NumItems)

	fmt.Println("Labels:")
	for i, label := range response.Labels {
		fmt.Printf("[%d] %d\n", i, label) // Print each label with its index.
	}
	fmt.Println()
}

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s file1 file2 file3 ...\n", os.Args[0])
		os.Exit(1)
	}

	for _, fileName := range os.Args[1:] {
		processFile(fileName) // Process each file specified in the command-line arguments.
	}
}