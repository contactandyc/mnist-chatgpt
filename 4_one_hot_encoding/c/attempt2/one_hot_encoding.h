#ifndef ONE_HOT_ENCODING_H
#define ONE_HOT_ENCODING_H

// Function to convert an unsigned char value to a one-hot encoded array of floats
float* oneHotEncoding(unsigned char input);

// Function to free the memory allocated for the one-hot encoded array
void freeOneHotEncoded(float* encoded);

#endif /* ONE_HOT_ENCODING_H */