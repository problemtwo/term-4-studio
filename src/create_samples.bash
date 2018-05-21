#!/bin/bash
# Made with help from https://www.youtube.com/watch?v=eay7CgPlCyo

mkdir create_samples_output
mkdir info
opencv_createsamples -img test.jpg -bg bg.txt -info info/info.lst -pngoutput create_samples_output -maxxangle 0.5 \
                          -maxyangle 0.5 -maxzangle 0.5 -num 1200
