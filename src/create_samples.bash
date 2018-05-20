#!/bin/bash
# Made with help from https://www.youtube.com/watch?v=eay7CgPlCyo

mkdir create_samples_output
opencv_createsamples -img ../training/images/IMG_0002.JPG -bg bg.txt -info info.lst -pngoutput create_samples_output -maxxangle 0.5 \
                                                          -maxyangle 0.5 -maxzangle 0.5 -num 1200
