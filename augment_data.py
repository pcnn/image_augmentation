import cv2
import numpy as np

def smooth_gaussian(im, ks):
    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    return cv2.GaussianBlur(im, ksize=ks, sigmaX=sigma_x, sigmaY=sigma_y)

def motion_blur(im, theta, ks):
    kernel = np.zeros((ks, ks), dtype='float32')

    if ks < 3:
        return im

    half_len = ks // 2
    x = np.linspace(-half_len, half_len, 2*half_len+1,dtype='int32')
    slope = np.arctan(theta*np.pi/180.)
    y = -np.round(slope*x).astype('int32')
    x += half_len
    y += half_len
    kernel[y, x] = 1.0

    kernel = np.divide(kernel, kernel.sum())
    im_res = cv2.filter2D(im, cv2.CV_8UC3, kernel)

    return im_res

def blur_median(im, ks):
    return cv2.medianBlur(im, ks)

def sharpen(im, ks=(3, 3), alpha=1):
    sigma_x = (ks[1] // 2.) / 3.
    sigma_y = (ks[0] // 2.) / 3.
    im_res = im.astype('float32') * 0.0039215
    im_coarse = cv2.GaussianBlur(im_res, ks, sigmaX=sigma_x, sigmaY=sigma_y)
    im_fine = im_res - im_coarse
    im_res += alpha * im_fine
    return np.clip(im_res * 255, 0, 255).astype('uint8')

def crop(im, shape, rand):
    dx = rand.randint(0, im.shape[0] - shape[0])
    dy = rand.randint(0, im.shape[1] - shape[1])
    im_res = im[dy:shape[0] + dy, dx:shape[1] + dx, :].copy()
    return im_res

def histeq(im):
    im_res = im.copy()
    im_res[:, :, 0] = cv2.equalizeHist(im_res[:, :, 0])
    im_res[:, :, 1] = cv2.equalizeHist(im_res[:, :, 1])
    im_res[:, :, 2] = cv2.equalizeHist(im_res[:, :, 2])
    return im_res

def normalize(im):
    return cv2.normalize(im,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def hsv(im, scale, p, component):
    if component != 1 and component != 2:
        raise Exception('componenet can be only 1 or 2')

    im_res = im.astype('float32')/255.
    im_res = cv2.cvtColor(im_res, cv2.COLOR_BGR2HSV)
    im_res[:, :, component] = np.power(im_res[:, :, component]*scale, p)
    im_res[:, :, component] = np.clip(im_res[:, :, component],0,1)
    im_res = (cv2.cvtColor(im_res, cv2.COLOR_HSV2BGR)*255).astype('uint8')
    return im_res

def resize(im, scale_x, scale_y, interpolation=cv2.INTER_NEAREST):
    im_res = cv2.resize(im, None, fx=scale_x, fy=scale_y, interpolation=interpolation)
    return im_res

def flip(im):
    return im[:, -1::-1, :].copy()

if __name__ == '__main__':
    # IMAGE IS TAKEN FROM http://www.drodd.com/images15/nature13.jpg
    im = cv2.imread('nature.jpg')

    im_aug = hsv(im, 1.4, 1, 1)
    cv2.imshow('src', im_aug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()