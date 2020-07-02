from small_img import colorize
from large_img import colorize_large
from original_img import colorize_orig
import os


if __name__ == '__main__':
    dir = 'data'  # can change to "own_choice" to process images downloaded from the website
    for file in os.listdir(dir):
        if '.jpg' in file or '.tif' in file:
            print('Coloring original image:' + file + '...')
            colorize_orig(dir + '/' + file, is_save=True, is_show=False)
        if '.jpg' in file:
            print('Aligning small image:' + file + '...')
            colorize('data/'+file, is_save=True, is_show=False)
        if '.tif' in file:
            print('Aligning large image:' + file + '...')
            colorize_large(dir + '/' + file, is_save=True, is_show=False)
