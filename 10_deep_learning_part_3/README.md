# My Commentary

This section is all about cleaning up the code generated by ChatGPT.

## Using AnotherCLibrary.com

AnotherCLibrary.com has a malloc/free replacement which keeps track of memory usage and is useful for finding common mistakes.

In attempt1
1. Copy ac_allocator.h/c and ac_common.h into the directory from [AnotherCLibrary github repo](https://github.com/contactandyc/another-c-library/tree/main/src).
2. Replace all calls to free and malloc with ac_free and ac_malloc
3. Replace all calls to calloc with ac_calloc.  ac_calloc takes one parameter.  To make the calls compatible, multiply the two parameters in calloc.
4. Add `#include "ac_allocator.h` to any c file which calls the ac_... methods
5. Add ac_allocator.c to the Makefile

Build the program and run it with ` 2>&1 | sort | uniq -c` added to the end of the command line.

```
% make clean; make
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o train_and_test
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c ac_allocator.c -o ac_allocator.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o

% ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte  ../../../data/t10k-labels-idx1-ubyte 2>&1 | sort | uniq -c 
   1 2504510512 byte(s) allocated in 2861277 allocations (114451080 byte(s) overhead)
   1 Epoch 1: loss = 0.86514
   1 Epoch 2: loss = 0.80352
   1 Epoch 3: loss = 0.73679
   1 Epoch 4: loss = 0.72054
   1 Epoch 5: loss = 0.70861
   1 Test Accuracy: 45.00%
3000 activation.c:18: 1024 
3000 activation.c:18: 512 
3100 activation.c:46: 40 
3000 activation.c:76: 256 
3100 activation.c:9: 1024 
3100 activation.c:9: 256 
3100 activation.c:9: 512 
   1 image_utils.c:40: 480000 
   1 image_utils.c:40: 80000 
70000 image_utils.c:52: 784 
576000 matrix.c:10: 1024 
798000 matrix.c:10: 256 
384000 matrix.c:10: 3136 
192000 matrix.c:10: 40 
768000 matrix.c:10: 512 
3000 matrix.c:59: 1024 
3000 matrix.c:59: 256 
3000 matrix.c:59: 512 
3000 matrix.c:79: 1024 
3000 matrix.c:79: 256 
3000 matrix.c:79: 512 
6000 matrix.c:7: 1024 
6000 matrix.c:7: 2048 
6000 matrix.c:7: 512 
3000 matrix.c:7: 80 
3100 matrix.c:98: 1024 
3100 matrix.c:98: 256 
3100 matrix.c:98: 40 
3100 matrix.c:98: 512 
   1 neural_network.c:10: 16 
   1 neural_network.c:19: 1024 
   1 neural_network.c:19: 256 
   1 neural_network.c:19: 40 
   1 neural_network.c:19: 512 
   1 neural_network.c:28: 1024 
   1 neural_network.c:28: 2048 
   1 neural_network.c:28: 512 
   1 neural_network.c:28: 80 
  64 neural_network.c:30: 1024 
  10 neural_network.c:30: 256 
 128 neural_network.c:30: 3136 
 256 neural_network.c:30: 512 
   4 neural_network.c:39: 48 
   1 neural_network.c:54: 1024 
   1 neural_network.c:54: 256 
   1 neural_network.c:54: 40 
   1 neural_network.c:54: 512 
```

This shows the frequency, line/location and the number of bytes allocated at the given location.  A lot of the allocations are happening in the matrix.c/h.  To improve this, new arrays and matrices are added to the layers that can be reused.

```c
typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    float* (*activation_function)(float*, int);
    float* (*activation_derivative)(float*, int);
    float* outputs;
} Layer;
```

becomes

```c
typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    float** gradients;
    float** transposed;
    float* output_error;
    void (*activation_function)(float*, int);
    void (*activation_derivative)(float*, int);
    float* outputs;
} Layer;
```

`gradients` which is a size x inputs_size array and is used by backward propagation.
`transposed` is used to transpose the `weights` matrix and is inputs_size x size.
`output_error` is used to track the difference between the output and target (or the next layer).

The activation functions just manipulate the inputs in place.  activation.c/h need changed.

The next change is to remove the allocations from matrix by passing a result array or matrix to the functions.

A new function

```c
void freeNeuralNetwork(NeuralNetwork *network);
```

is introduced in neural_network.h

In attempt2, the changes have been made

```bash
% make clean;make
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o train_and_test
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c ac_allocator.c -o ac_allocator.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o
% ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
Epoch 1: loss = 0.90803
Epoch 2: loss = 0.86914
Epoch 3: loss = 0.84207
Epoch 4: loss = 0.84257
Epoch 5: loss = 0.83887
Test Accuracy: 29.00%
```

The memory leaks are now gone and due to the way it was implemented, memory is only allocated at the beginning and freed at the end.

For attempt3, I attempt the relu and reluDerivative again.

In main.c
```c
addDenseLayer(network, 128, sigmoid, sigmoidDerivative);
addDenseLayer(network, 256, sigmoid, sigmoidDerivative);
addDenseLayer(network, 64, sigmoid, sigmoidDerivative);
```

becomes

```c
addDenseLayer(network, 128, relu, reluDerivative);
```

Running with this change

```bash
% make clean;make                                                                                                                                                       
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o train_and_test
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c ac_allocator.c -o ac_allocator.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o
% ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
Epoch 1: loss = nan
Epoch 2: loss = nan
Epoch 3: loss = nan
Epoch 4: loss = nan
Epoch 5: loss = nan
Test Accuracy: 8.00%
```

In order to find the cause of the nan and infinites, I'm using the following macro defined functions.

```c
#ifdef DEBUG_NETWORK
void printVector(float *o, int num) {
    printf("[");
    for (int i = 0; i < num; i++) {
        printf("%.4f ", o[i]);
    }
    printf("]\n");
}

void testVector(float* o, int num) {
    for( int i=0; i<num; i++ ) {
        if(isnan(o[i]) || isinf(o[i])) {
            printVector(o, num);
            abort();
        }
    }
}

void testMatrix(float** m, int a, int b) {
    for( int i=0; i<a; i++ )
        testVector(m[i], b);
}

void testNetwork(NeuralNetwork* network) {
    for( int i=0; i<network->num_layers; i++ ) {
        Layer *layer = network->layers[i];
        testVector(layer->biases, layer->size);
        testMatrix(layer->weights, layer->size, layer->inputs_size);
    }
}
#else
#define printVector(a, b) ;
#define testVector(a, b) ;
#define testMatrix(a, b, c) ;
#define testNetwork(a) ;
#endif
```

If `DEBUG_NETWORK` is defined, the functions are tested for nan and infinite.  If it is not, the function calls end up being a `noop`.

In attempt4, this is added along with a few calls in key places to check that everything is okay.  If there is a problem, it is aborted (and can be debugged).

```bash
% make clean;make                                                                                                                                                       
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o train_and_test
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -c ac_allocator.c -o ac_allocator.o
gcc -Wall -Wextra -std=c99 -g -D_AC_DEBUG_MEMORY_=NULL -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o ac_allocator.o
ac@MacBook-Pro attempt4 % lldb ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
(lldb) target create "./train_and_test"
Current executable set to '/Users/ac/code/neural/10_deep_learning_part_3/c/attempt4/train_and_test' (x86_64).
(lldb) settings set -- target.run-args  "../../../data/train-images-idx3-ubyte" "../../../data/train-labels-idx1-ubyte" "../../../data/t10k-images-idx3-ubyte" "../../../data/t10k-labels-idx1-ubyte"
(lldb) r
Process 85043 launched: '/Users/ac/code/neural/10_deep_learning_part_3/c/attempt4/train_and_test' (x86_64)
[0.2987 -2565361013595570176.0000 426018596364222464.0000 739625666525265920.0000 641950757230215168.0000 1086494853212143616.0000 14949610535277453584033464188928.0000 301840934084989905495952195584.0000 134016999815615710053329969086464.0000 -32950366082537649781390761263104.0000 18751401767175311744515855876096.0000 -37073451111662848708376463933440.0000 -62199920089038212148003988307968.0000 -408840215512570604376251317935407104.0000 -101110276186970420741333557251821010944.0000 -6788877436510277995127564082327060480.0000 21516759133252795250197498167296.0000 -inf -48085711145037416533309383507968.0000 inf -291482321507083642176585650314577707008.0000 -19115222789371606773279711607914496.0000 -66473409997233306319873235746816.0000 inf 16158310285763814823083970134016.0000 -35052127141267188728267737661440.0000 -inf -64412118939241186699248723296256.0000 inf -inf 5996095039862942897163322385235968.0000 -266600267053515845933895641381918998528.0000 -1231196613461141180069187324211101696.0000 -5946079883109800532467181944832.0000 -inf 64989088980314779463430446914732032.0000 inf -44347161240615238854214221299712.0000 -2788312147308871180793669484544.0000 -inf -25821581301830549832985699493542363136.0000 -17836589465585627628578071904256.0000 20751658361639354557958088904956444672.0000 -177344278407110085966047930537599827968.0000 28508673489356293793927944732672.0000 -35140632600521175730147976806400.0000 682115237515711255817402449920.0000 inf -26310389375337083624484077830144.0000 -50153467538312835190760224587776.0000 -50493175693624545988852660043776.0000 inf 34609711066172658264750614904832.0000 -30514022713971566064122103595008.0000 -3094588970236413927346074025984.0000 inf -31741764255042677327303406518272.0000 inf -inf -inf -16818618314882407590533435228160.0000 inf 140972977081863763829025162800398336.0000 inf -17998199563486018387884705859372056576.0000 -167133452808212791138586029270761472.0000 35892702935051797310474339483648.0000 -447096127993801996765825965934772224.0000 -37275598015812250081937433034752.0000 -3638063416510601493319827443391922176.0000 -7898659495698894239185416224768.0000 19429310547301673898119868186624.0000 -12225155965380811441848830656512.0000 inf -1529799476704827013496819931544551424.0000 109493827654953405846038338045634150400.0000 182578864089668527327412356885751267328.0000 -374600735115918585448812583726350336.0000 15615439728614226362923812388864.0000 -49771877356906395179539268370432.0000 -33003708725402325679095496572928.0000 -5145409631958885993542637019411251200.0000 inf -58735274089714481771763654983680.0000 132601244158111194049823548551200768.0000 -69404822996041416059734168961024.0000 inf -48116480724998248075064005099520.0000 -14957504820879537112544295518208.0000 inf -404653463730138099301413289984.0000 -64508238213307106635671261937664.0000 -5212970341134014796116910931968.0000 -51894489968172647250421576892416.0000 -56488376727853438084372097978672873472.0000 -464706171675847531082742979775954944.0000 -56275743523928189466924132859904.0000 -59058971231471216421066281844736.0000 -1359177033496932935157542608946855936.0000 211106661387083107677136460289820065792.0000 -15844580737389802791326795694080.0000 -4207191526478024671561288163262464.0000 inf inf -55860554876143019681679742599168.0000 -15749151615669496978960875986238832640.0000 14021173639700712597296741613568.0000 -192373584488235181053533691635040256000.0000 -inf -172960681784292057740661059011655761920.0000 -inf -inf -3078239438573360083285298509548355584.0000 -19536570072795342642387324239872.0000 -5032423401646327014273717656092672.0000 31056285945475006362253022547935232.0000 -9481527831985081280953357172736.0000 -inf 108038879697837392693862057103065088.0000 -53969480573552639848853859729408.0000 -5215819779290846477081693388800.0000 inf -12291014722403489682201286105708363776.0000 -35896581169081121040866796896256.0000 -6177535380367029168425140224000.0000 -29648335113061922404698505084928.0000 -22035090914115845967369469952000.0000 -31449172774624627086665969565696.0000 ]
Process 85043 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
    frame #0: 0x00007ff8055121f2 libsystem_kernel.dylib`__pthread_kill + 10
libsystem_kernel.dylib`:
->  0x7ff8055121f2 <+10>: jae    0x7ff8055121fc            ; <+20>
    0x7ff8055121f4 <+12>: movq   %rax, %rdi
    0x7ff8055121f7 <+15>: jmp    0x7ff80550bcdb            ; cerror_nocancel
    0x7ff8055121fc <+20>: retq   
Target 0: (train_and_test) stopped.
(lldb) bt
* thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
  * frame #0: 0x00007ff8055121f2 libsystem_kernel.dylib`__pthread_kill + 10
    frame #1: 0x00007ff805549ee6 libsystem_pthread.dylib`pthread_kill + 263
    frame #2: 0x00007ff805470b45 libsystem_c.dylib`abort + 123
    frame #3: 0x00000001000051c9 train_and_test`testVector(o=0x0000000100670ee8, num=128) at neural_network.c:73:13
    frame #4: 0x0000000100005752 train_and_test`backwardPass(network=0x0000600001704068, input=0x000000010086a828, error=0x0000000100670ee8, learning_rate=0.00999999977) at neural_network.c:182:13
    frame #5: 0x0000000100005903 train_and_test`train(network=0x0000600001704068, data=0x00007ff7bfeff1e0, epochs=5, learning_rate=0.00999999977) at neural_network.c:216:13
    frame #6: 0x00000001000036fe train_and_test`main(argc=5, argv=0x00007ff7bfeff530) at main.c:57:5
    frame #7: 0x00007ff8051f041f dyld`start + 1903
(lldb) up
frame #1: 0x00007ff805549ee6 libsystem_pthread.dylib`pthread_kill + 263
libsystem_pthread.dylib`pthread_kill:
->  0x7ff805549ee6 <+263>: movl   %eax, %r15d
    0x7ff805549ee9 <+266>: cmpl   $-0x1, %eax
    0x7ff805549eec <+269>: jne    0x7ff805549f36            ; <+343>
    0x7ff805549eee <+271>: movq   %gs:0x8, %rax
(lldb) 
frame #2: 0x00007ff805470b45 libsystem_c.dylib`abort + 123
libsystem_c.dylib`abort:
->  0x7ff805470b45 <+123>: movl   $0x2710, %edi             ; imm = 0x2710 
    0x7ff805470b4a <+128>: callq  0x7ff805444c3c            ; usleep$NOCANCEL
    0x7ff805470b4f <+133>: callq  0x7ff805470b54            ; __abort

libsystem_c.dylib`:
    0x7ff805470b54 <+0>:   pushq  %rbp
(lldb) 
frame #3: 0x00000001000051c9 train_and_test`testVector(o=0x0000000100670ee8, num=128) at neural_network.c:73:13
   70  	    for( int i=0; i<num; i++ ) {
   71  	        if(isnan(o[i]) || isinf(o[i])) {
   72  	            printVector(o, num);
-> 73  	            abort();
   74  	        }
   75  	    }
   76  	}
(lldb) 
frame #4: 0x0000000100005752 train_and_test`backwardPass(network=0x0000600001704068, input=0x000000010086a828, error=0x0000000100670ee8, learning_rate=0.00999999977) at neural_network.c:182:13
   179 	            testVector(output_error, layer->inputs_size);
   180 	
   181 	            output_error = matrixVectorMultiply(transposed, output_error, layer->output_error, layer->inputs_size, layer->size);
-> 182 	            testVector(output_error, layer->inputs_size);
   183 	
   184 	            layer->activation_derivative(layer_inputs, layer->inputs_size);
   185 	            testVector(layer_inputs, layer->inputs_size);
(lldb)
```

Corruption is happening in `matrixVectorMultiply`.  After some investigation, the problem is that two very small numbers are multiplied together causing a `-inf`.

The next section will explore why this is happening and how to fix it.  The working version in the c directory switches back to sigmoid activation and comments out the `#define DEBUG_NETWORK`
