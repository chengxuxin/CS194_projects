# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import os


def colorize_orig(imname, is_save, is_show):
    im = skio.imread(imname)
    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    # create a color image
    im_out = np.dstack([r, g, b])

    if is_save:
        # save the image
        if not os.path.exists('original'):
            os.mkdir('original')
        skio.imsave('original/'+'o_' + imname.split('/')[1].split('.')[0]+'.jpg', im_out)
    if is_show:
        # display the image
        skio.imshow(im_out)
        skio.show()


if __name__ == '__main__':
    dir = 'data'  # can change to "own_choice" to process images downloaded from the website
    for file in os.listdir(dir):
        if '.jpg' in file or '.tif' in file:
            print(file+'---------------')
            colorize_orig(dir+'/'+file, is_save=True, is_show=False)

