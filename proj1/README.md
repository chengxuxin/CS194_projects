# CS194 Project 1 

## Colorizing the Prokudin-Gorskii photo collection

##### Xuxin Cheng



### Files

small_img.py is a simple version to align small images ending with .jpg.

large_img.py has implementation of image pyramid used to process large images ending with .tif.

original_img.py is used to generate 3-channel pictures without any alignment.

---

### How to run

Make a folder called 'data'.

Put all the .jpg and .tif files in it.

Run main.py to process all the images.

Original images without alignments will be saved in 'original' folder.

Aligned images will be saved in 'output' folder.

--------

### Notes

The offset of each channel of each image is saved to .csv files.

There is also implementation of Canny Edge Detection in large_img.py. When encountered with emir_ed.tif, the code will automatically perform edge detection before alignment.

