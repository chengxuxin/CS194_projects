# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
from skimage.transform import resize
import skimage.io as skio
import os
from small_img import align
import cv2 as cv


def pyramid(resize_factor, r, g, b, xy_g, xy_r):
    b_rsz = resize(b, (b.shape[0] // resize_factor, b.shape[1] // resize_factor))
    g_rsz = resize(g, (g.shape[0] // resize_factor, g.shape[1] // resize_factor))
    r_rsz = resize(r, (r.shape[0] // resize_factor, r.shape[1] // resize_factor))

    # roll the image according to last time align result
    g_rsz = np.roll(np.roll(g_rsz, 2*xy_g[0], axis=0), 2*xy_g[1], axis=1)
    r_rsz = np.roll(np.roll(r_rsz, 2*xy_r[0], axis=0), 2*xy_r[1], axis=1)

    if resize_factor == 16:
        steps = 16
    else:
        steps = 4
    mov_range = range(-steps, steps+1)
    x_g, y_g = align(g_rsz, b_rsz, mov_range)
    x_r, y_r = align(r_rsz, b_rsz, mov_range)
    if resize_factor == 1:
        return (x_g + 2*xy_g[0], y_g+2*xy_g[1]), (x_r+2*xy_r[0], y_r+2*xy_r[1])
    return pyramid(int(resize_factor/2), r, g, b, (x_g + 2*xy_g[0], y_g+2*xy_g[1]), (x_r+2*xy_r[0], y_r+2*xy_r[1]))


def colorize_large(imname, is_save, is_show):
    # read in the image
    im = skio.imread(imname)
    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)
    # separate color channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]
    # clip boarders
    num_clip_x = int(0.1 * b.shape[0])
    num_clip_y = int(0.1 * b.shape[1])
    b_clip = b[num_clip_x: b.shape[0]-num_clip_x, num_clip_y: b.shape[1]-num_clip_y]
    g_clip = g[num_clip_x: g.shape[0]-num_clip_x, num_clip_y: g.shape[1]-num_clip_y]
    r_clip = r[num_clip_x: r.shape[0]-num_clip_x, num_clip_y: r.shape[1]-num_clip_y]
    # edge detection for emir_ed.tif
    if imname == 'data/emir_ed.tif':
        print('edge detection in progress...')
        b_clip = cv.Canny(np.uint8(np.array(b_clip)*255), 50, 150)
        g_clip = cv.Canny(np.uint8(np.array(g_clip)*255), 50, 150)
        r_clip = cv.Canny(np.uint8(np.array(r_clip)*255), 50, 150)
        b_clip = sk.img_as_float(b_clip)
        g_clip = sk.img_as_float(g_clip)
        r_clip = sk.img_as_float(r_clip)
    # align 3 channels
    xy_g, xy_r = (0, 0), (0, 0)
    down_factor = 16
    xy_g, xy_r = pyramid(down_factor, r_clip, g_clip, b_clip, xy_g, xy_r)
    # xy_g, xy_r = (60, 6), (134, 8)
    g_final = np.roll(np.roll(g, xy_g[0], axis=0), xy_g[1], axis=1)
    r_final = np.roll(np.roll(r, xy_r[0], axis=0), xy_r[1], axis=1)
    # create a color image and show and save
    im_out = np.dstack([r_final, g_final, b])
    if is_save:
        if not os.path.exists('output'):
            os.mkdir('output')
        skio.imsave('output/'+imname.split('/')[1].split('.')[0]+'.jpg', im_out)
        f = open('offset_large.csv', 'a')
        f.write('name, green_x, green_y, red_x, red_y\n')
        f.write('%s, %s, %s, %s, %s\n' % (imname.split('/')[1], xy_g[0], xy_g[1], xy_r[0], xy_r[1]))
    if is_show:
        skio.imshow(im_out)
        skio.show()


if __name__ == '__main__':
    dir = 'data'  # # can change to "own_choice" to process images downloaded from the website
    for file in os.listdir(dir):
        if '.tif' in file:
            print(file+'---------------')
            colorize_large(dir + '/'+file, is_save=True, is_show=False)

