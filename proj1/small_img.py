# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import os


def align1(im, base_im, mr):
    score = np.empty((len(mr), len(mr)))
    for i in range(len(mr)):
        for j in range(len(mr)):
            im_temp = np.roll(np.roll(im, mr[i], axis=0), mr[j], axis=1)
            score[i, j] = SSD(im_temp, base_im)
    pos = np.argmin(score)
    return mr[pos//len(mr)], mr[pos % len(mr)]


def align(im, base_im, mr):
    score_min = np.inf
    pos = [0, 0]
    for i in range(len(mr)):
        for j in range(len(mr)):
            im_temp = np.roll(np.roll(im, mr[i], axis=0), mr[j], axis=1)
            score = SSD(im_temp, base_im)
            if score < score_min:
                score_min = score
                pos = (mr[i], mr[j])
    return pos


def SSD(im1, im2):
    # Sum of Squared Difference
    return np.sum(np.square(im1-im2))
    # return np.linalg.norm(im1.flatten() - im2.flatten(), ord=2)


def colorize(imname, is_save, is_show):
    im = skio.imread(imname)
    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int)

    # separate color channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    num_clip = 30
    b_clip = b[num_clip: b.shape[0] - num_clip - 1, num_clip: b.shape[1] - num_clip - 1]
    g_clip = g[num_clip: g.shape[0] - num_clip - 1, num_clip: g.shape[1] - num_clip - 1]
    r_clip = r[num_clip: r.shape[0] - num_clip - 1, num_clip: r.shape[1] - num_clip - 1]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    mov_range = range(-15, 15)
    xy_g = align(g_clip, b_clip, mov_range)
    g = np.roll(np.roll(g, xy_g[0], axis=0), xy_g[1], axis=1)
    xy_r = align(r_clip, b_clip, mov_range)
    r = np.roll(np.roll(r, xy_r[0], axis=0), xy_r[1], axis=1)

    # create a color image
    im_out = np.dstack([r, g, b])

    if is_save:
        # save the image
        if not os.path.exists('output'):
            os.mkdir('output')
        skio.imsave('output/'+imname.split('/')[1], im_out)
        f = open('offset_small.csv', 'a')
        f.write('name, green_x, green_y, red_x, red_y\n')
        f.write('%s, %s, %s, %s, %s\n' % (imname.split('/')[1], xy_g[0], xy_g[1], xy_r[0], xy_r[1]))
    if is_show:
        # display the image
        skio.imshow(im_out)
        skio.show()


if __name__ == '__main__':
    for file in os.listdir('data'):
        if '.jpg' in file:
            print(file+'---------------')
            colorize('data/'+file, is_save=True, is_show=False)

