![alt tag](https://raw.github.com/alfonsoros88/ScaRF/master/doc/logo.png)

Simple CUDA accelerated Random Forest


###Compilation

This library was intended to be as general as possible, so it could be adapted 
to different problems. Therefore, it is essentially a template library. In 
principle you just need to include the header files in your code and the rest 
would be handled by the compiler.

###Documentation

We use Doxygen for the documentation. There is a cmake command to build the 
documentation. Just follow the standar cmake building procidure as follows:

```bash
    mkdir build
    cd build
    cmake ..
    make doc
```
