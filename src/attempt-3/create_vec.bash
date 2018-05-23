#!/bin/bash
# Made with help from https://www.youtube.com/watch?v=eay7CgPlCyo

opencv_createsamples -info info/info.lst -num 1400 -w 100 -h 100 -vec positives.vec
