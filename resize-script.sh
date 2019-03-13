#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

if [ `ls originalGrimace/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  # echo hi
  for file in originalGrimace/*/*.jpg; do
    convert "$file" -resize 64x71 "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls originalGrimace/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  # echo hi
  for file in originalGrimace/*/*.png; do
    convert "$file" -resize 64x71 "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

if [ `ls originalGrimace/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  # echo hi
  for file in originalGrimace/*/*.jpg; do
    convert "$file" -resize 64x71 "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls originalGrimace/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  # echo hi
  for file in originalGrimace/*/*.png; do
    convert "$file" -resize 64x71 "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi
