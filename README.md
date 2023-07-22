Virtually everything in this REPO is generated by ChatGPT using the 3.5 version!

The directory structure is as follows

`data` a place for the data used within the project<br/>
`1_reading_mnist_labels` shows how to use ChatGPT to generate a program to read the mnist label data and print it to the screen.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`c` a c version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python` a python version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`go` a go version of `file_reader`<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`psuedocode` a non-running psuedocode version of `file_reader`<br/>

`1_reading_mnist_labels` asks ChatGPT to convert the following file format to code.  In addition to this, it shows how to generate the program in multiple languages and deals with various issues.

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

