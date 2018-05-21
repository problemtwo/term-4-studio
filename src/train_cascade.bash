#!/bin/bash
# Made with help from https://www.youtube.com/watch?v=eay7CgPlCyo

if [ ! -d ./create_samples_output/ ]; then
 mkdir create_samples_output
fi
opencv_traincascade -data info/pos -vec positives.vec -bg bg.txt -numPos 900 -numNeg 450 -numStages 10 -w 50 -h 71
