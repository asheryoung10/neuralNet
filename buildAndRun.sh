clang ./source/main.c ./source/idxReader.c -o program
./program ./data/t10k-images.idx3-ubyte ./data/t10k-labels.idx1-ubyte > output.txt