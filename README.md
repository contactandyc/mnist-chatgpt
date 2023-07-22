Virtually everything in this REPO is generated by ChatGPT using the 3.5 version!

The directory structure is as follows

`data` a place for the data used within the project<br/>
`1_reading_mnist_labels` shows how to use ChatGPT to generate a program to read the mnist label data and print it to the screen.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`c` a c version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python` a python version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`go` a go version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`psuedocode` a non-running psuedocode version of `file_reader`<br/>
`2_reading_mnist_images` shows how to use ChatGPT to generate a program to read the mnist image data and print it to the screen.</br>

## details...

### data
Comes from the website http://yann.lecun.com/exdb/mnist/ and is needed for this project

### 1_reading_mnist_labels
Asks ChatGPT to convert the following file format to code.  In addition to this, it shows how to generate the program in multiple languages and deals with various issues.

```
generate a function in c to read a file with the following format and return a structure 
containing the response

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
```

yielding

```bash
File: ../../data/t10k-labels-idx1-ubyte
Magic Number: 2049
Number of Items: 10000
Labels:
[0] 7
[1] 2
[2] 1
[3] 0
[4] 4
[5] 1
[6] 4
...
```


### 2_reading_mnist_images
Asks ChatGPT to convert the following file format to code in C.  The code can easily be converted to other languages using the technique in `1_reading_mnist_labels`.

```
generate a function in c to read a file with the following format and return a structure
containing the response.  Allow the names of the file to be passed to a function which are 
derived from the command line.  Also, generate a makefile for this project.

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
```

yielding

```bash
% ./file_reader ../../data/t10k-images-idx3-ubyte
Magic Number: 0x00000803
Number of Images: 10000
Number of Rows: 28
Number of Columns: 28

Image 1:
Pixel Values (First 200): 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Image 2:
Pixel Values (First 200): 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 116 125 171 255 255 150 93 0 0 0 0 0 0 0 0 0 
 0 0 0 0 0 0 0 0 0 0 0 169 253 253 253 253 253 253 218 30 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
 0 0 0 0 169 253 253 253 213 142 176 253 253 122 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 52 
 250 253 210 32 12 0 6 206 253 140 0 0 0 0 0 0 0 0 0 0 0 0 0 0 

...
```

