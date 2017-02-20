__author__ = "Hamed H. Aghdam"
__credits__ = "Hamed H. Aghdam"
__version__ = "1.0.0"

import cv2
import numpy as np

def smooth_gaussian(im, ks):
    '''
    Smmothes the image using a Gaussian kernel. The variance of Gaussian filter is computed based on the kernel size.
    :param im: Image to be smoothed
    :param ks: (tuple, list or int) Size of Guassian filter
    :return:
    '''

    if isinstance(ks, int) or isinstance(ks, float):
        ks = (ks, ks)

    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    return cv2.GaussianBlur(im, ksize=ks, sigmaX=sigma_x, sigmaY=sigma_y)

def motion_blur(im, theta, ks):
    '''
    Simulated motion blur effect on the image.
    :param im: Input image.
    :param theta: (float) Direction of blur in Degrees
    :param ks: Size of filter.
    :return: Image after applying motion blur effect.
    '''

    if ks < 3:
        return im

    theta = theta * np.pi / 180.

    # Creating a filter where all elements except elementing lying of line with oreinetaion theta are zero
    kernel = np.zeros((ks, ks), dtype='float32')
    half_len = ks // 2
    x = np.linspace(-half_len, half_len, 2*half_len+1,dtype='int32')
    y = -np.round(np.sin(theta)*x/(np.cos(theta)+1e-4)).astype('int32')
    ind = np.where(np.abs(y) <= half_len)
    # print x, y,theta
    x += half_len
    y += half_len
    kernel[y[ind], x[ind]] = 1.0

    y = np.linspace(-half_len, half_len, 2 * half_len + 1, dtype='int32')
    x = -np.round(np.cos(theta) * y/ (np.sin(theta)+1e-4)).astype('int32')
    ind = np.where(np.abs(x) <= half_len)
    # print x, y, theta
    x += half_len
    y += half_len
    kernel[y[ind], x[ind]] = 1.0

    #Normalizing filter
    kernel = np.divide(kernel, kernel.sum())
    im_res = cv2.filter2D(im, cv2.CV_8UC3, kernel)
    # np.set_printoptions(2,linewidth=120)
    # print kernel
    # import matplotlib.pyplot as plt
    # plt.clf()
    # plt.imshow(kernel)
    # plt.show()

    return im_res

def blur_median(im, ks):
    '''
    Smoothing the image using median filtering.
    :param im: Input image
    :param ks: (int) Window size
    :return: Smoothed image.
    '''
    return cv2.medianBlur(im, ks)

def sharpen(im, ks=(3, 3), alpha=1):
    '''
    Sharpens the input image.
    :param im: Input image.
    :param ks: (tuple or list) Kernel size
    :param alpha: Strength of fine image. [Default=1]
    :return: Sharpenned image.
    '''
    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    im_res = im.astype('float32') * 0.0039215
    im_coarse = cv2.GaussianBlur(im_res, ks, sigmaX=sigma_x, sigmaY=sigma_y)
    im_fine = im_res - im_coarse
    im_res += alpha * im_fine
    return np.clip(im_res * 255, 0, 255).astype('uint8')

def crop(im, shape, rand):
    '''
    Randomly crops the image.
    :param im: Input image
    :param shape: (list, tuple) shape of cropped image.
    :param rand: An instance of numpy.random.RandomState objects
    :return: Randomlly cropped image.
    '''
    dx = rand.randint(0, im.shape[0] - shape[0])
    dy = rand.randint(0, im.shape[1] - shape[1])
    im_res = im[dy:shape[0] + dy, dx:shape[1] + dx, :].copy()
    return im_res

def histeq(im):
    '''
    Applies histogram equalization on each channel of the image individually.
    :param im: Input image
    :return:
    '''
    im_res = im.copy()
    im_res[:, :, 0] = cv2.equalizeHist(im_res[:, :, 0])
    im_res[:, :, 1] = cv2.equalizeHist(im_res[:, :, 1])
    im_res[:, :, 2] = cv2.equalizeHist(im_res[:, :, 2])
    return im_res

def normalize(im):
    '''
    Normalizes each channel such that min(channel)=0 and max(channel)=255.
    :param im:Input image
    :return:
    '''

    return cv2.normalize(im,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    #implementation using numpy
    # im = im.astype('float32')
    # im_mn = np.min(im, axis=(0,1), keepdims=True)
    # im_mx = np.max(im, axis=(0,1), keepdims=True)
    # im_res = np.clip(((im-im_mn)/(im_mx-im_mn))*255,0,255)
    # return im_res.astype('uint8')

def hsv(im, scale, p=1, channel=2):
    '''
    Scales hue, saturation or value of image (res = (scale*HSV[:,:,channel])^p).
    :param im: Input image.
    :param scale: Scale
    :param p: power
    :param channel: H=0, S=1, V=2
    :return:
    '''
    if channel != 1 and channel != 2:
        raise Exception('componenet can be only 1 or 2')

    im_res = im.astype('float32')/255.
    im_res = cv2.cvtColor(im_res, cv2.COLOR_BGR2HSV)
    im_res[:, :, channel] = np.power(im_res[:, :, channel] * scale, p)
    im_res[:, :, channel] = np.clip(im_res[:, :, channel], 0, 1)
    im_res = (cv2.cvtColor(im_res, cv2.COLOR_HSV2BGR)*255).astype('uint8')
    return im_res

def resize(im, scale_x, scale_y, interpolation=cv2.INTER_NEAREST):
    im_res = cv2.resize(im, None, fx=scale_x, fy=scale_y, interpolation=interpolation)
    return im_res

def flip(im):
    return im[:, -1::-1, :].copy()

def power_law(im, c, gamma):
    '''
    Applies power-law transofrmation on image (c*im^gamma).
    :param im: Input image
    :param c: scale constant
    :param gamma: power
    :return:
    '''
    im = im.astype('float32')
    im = np.clip(c*np.power(im, gamma), 0,255)
    return im.astype('uint8')

if __name__ == '__main__':
    # IMAGE IS TAKEN FROM http://www.drodd.com/images15/nature13.jpg
    im = cv2.imread('examples/nature.jpg')
    im = cv2.resize(im, fx=0.5, fy=0.5, dsize=None)

    im_aug = motion_blur(im, 35,7)
    # im_aug = normalize(im)
    # im_aug = hsv(im, scale=1.4, p=1, component=1)
    # im_aug = power_law(im, c=1, gamma=0.9)
    # im_aug = histeq(im)

    cv2.imshow('src', im_aug)
    cv2.waitKey(0)

    cv2.destroyAllWindows()