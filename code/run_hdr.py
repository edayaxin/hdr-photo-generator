import os
import numpy as np
import scipy
from scipy.sparse import linalg
import cv2
import math
import matplotlib.pyplot as plt
import time
def weight_func(z):
    # weighting function
    return 127.5 - np.abs(1.0 * z - 127.5)

def select_Z(img_set, N, channel):
    """
    TODO 1
    Select a subset of pixels to estimate g()
    :param img_set:a list of total sequence
    :param N:number of pixels to select
    :param channel:the channel of images to select. Should be [0,1,2]
    :return:Z
    """
    rows = img_set[0].shape[0]
    cols = img_set[0].shape[1]

    inner_padding = 30

    a = np.sqrt(N)
    a = int(np.rint(a))

    nx = int(np.rint((rows - inner_padding*2)/a))
    ny = int(np.rint((cols - inner_padding*2)/a))
    Z = np.zeros((N, len(img_set)), dtype=np.float32)
    for i in range(0, N):
        for j in range(0, len(img_set)):
            mx = int(np.rint(i/a))
            my = int(np.rint(i%a))
            Z[i][j] = img_set[j][inner_padding + mx*nx][inner_padding + my*ny][channel]
            
    return Z

def radiance_map_construction(Z, exposure, lam):
    # min / max / mid value for Z
    Zmin = 0.0
    Zmax = 255.0
    Zmid = 128.0

    # size of each variable
    N, F = Z.shape
    n = 256

    # construct A and b
    I = np.array([])
    J = np.array([])
    S = np.array([])
    B = np.array([])
    """
    TODO 2
    Construct matrix A & b

    add your code here
    """
    for i in range(0, N): 
        for j in range(0, F):
            I = np.append(I, [i*F + j])
            J = np.append(J, [Z[i][j]])
            S = np.append(S, [weight_func(Z[i][j])])
            I = np.append(I, [i*F + j])
            J = np.append(J, [n + i])
            S = np.append(S, [(-1)*weight_func(Z[i][j])])
            B = np.append(B, [weight_func(Z[i][j]) * np.log(exposure[j])])

    for i in range(1, n-1):
        for k in range(0, 3):
            I = np.append(I, [F*N + i - 1])

        J = np.append(J, [i-1])
        J = np.append(J, [i+1])
        for k in range(0, 2):
            S = np.append(S, [np.sqrt(lam)*weight_func(i)])
        J = np.append(J, [i])
        S = np.append(S, [np.sqrt(lam)*weight_func(i)*(-2)])
        B = np.append(B, [0])

    # set g(128) = 0; 
    I = np.append(I, [F*N + n - 2])
    J = np.append(J, [128])
    S = np.append(S, [1])
    B = np.append(B, [0])

    A = scipy.sparse.coo_matrix((S, (I, J)), shape=(N*F + n - 1, n + N))

    # solve Ax=B with least square
    x = scipy.sparse.linalg.lsqr(A, B)
    y = x[0] 
    # the first 256 elements belong to g
    g = y[0:n]

    return g

def bFilter(img, d, sigS, sigR):
    w = img.shape[0]
    h = img.shape[1]
    padding = (d-1)/2
    filtImg = np.zeros(shape=(img.shape), dtype=np.float32)
    for i in range(padding, w-padding):
        for j in range(padding, h-padding):
            Wij = 0
            UpWij = 0
            for ki in range(-padding, padding+1):
                for kj in range(-padding, padding+1):
                    gs = np.exp(-np.power(np.power(ki, 2) - np.power(kj, 2), 2)/np.power(sigS, 2))
                    sub = np.abs(img[i,j] - img[i+ki, j+kj])
                    gr = np.exp(-np.power(sub,2)/np.power(sigR, 2))
                    Wij = gs*gr + Wij
                    UpWij = gs*gr*img[i+ki,j+kj] + UpWij
            filtImg[i][j] = UpWij/Wij
    return filtImg



def tone_mapping(hdr, g_set):
    """
    Tone mapping
    :param hdr:input a 3 channel hdr image (radiance map)
    :param g_set:
    :return:
    """
    """
    TODO 4
    Follow spec to do tone mapping locally

    add your code here
    """
    # compute intensity by averaging color channels
    r = hdr[:,:,0]
    g = hdr[:,:,1]
    b = hdr[:,:,2]
    print "img max", np.nanmax(hdr)
    print "before imgMin"
    imgMin = np.nanmax(hdr)/1000
    print "imgMin is:", imgMin
    img_I = np.zeros(shape=(hdr.shape), dtype=np.float32)
    # compute chrominance channels
    img_chrom = np.zeros(shape=(hdr.shape), dtype=np.float32)
    for i in range(0, hdr.shape[0]):
        for j in range(0, hdr.shape[1]): 
            for cha in range(0, hdr.shape[2]):
                img_I[i][j][cha] = (r[i][j] + g[i][j] + b[i][j]) / 3
                if img_I[i][j][cha] == 0:
                    img_I[i][j][cha] = imgMin
                img_chrom[i][j][cha] = hdr[i][j][cha]/img_I[i][j][cha]
    # compute log intensity
    img_L = np.log2(img_I)

    # bilateral filter
    # Note: change d and sigmaColor and sigmaSpace to produce best results
    img_B = cv2.bilateralFilter(img_L, d=3, sigmaColor=10, sigmaSpace=3)
    #img_B = bFilter(img_L, 3, 10, 10)
    # compute detail layer
    img_D = img_L - img_B

    # apply offset and scale to base
    o = np.nanmin(img_B)
    maxB = np.nanmax(img_B)
    # adjust to produce good result
    dR = 5
    s = dR / (maxB - o)
    img_Bp = np.subtract(img_B, o)*s

    # reconstruct log intensity
    img_O = np.power(2, img_Bp+img_D)

    # push back colors
    img_chromp = img_O * img_chrom

    result = np.power(img_chromp, 0.2)
    result = np.log2(result)

    return result

start = time.clock()
# path to data
dataPath = './data/'

# file names and exposure time for case 1
# # please change according to your data
# imgList = ['case1/belg%03d.jpg' % i for i in range(1,10)]
# exposure = [1.0/1000, 1.0/500, 1.0/250, 1.0/125, 1.0/60, 1.0/30, 1.0/15, 1.0/8, 1.0/4]
# imgList = ['case2/memorial00%d.png'% i for i in range(61,77)]
# exposure = [32, 16, 8, 4, 2, 1.0, 1/2.0, 1/4.0, 1/8.0, 1/16.0, 1/32.0, 1/64.0, 1/128.0, 1/256.0, 1/512.0, 1/1024.0]
# imgList = ['case3/%d.29.png'%(np.power(2, i)*3) for i in range(0, 8)]
# exposure = [0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192, 0.384]
# imgList = ['case4/IMG_09%d.jpg' % i for i in range(13,22)]
# exposure = [1.0, 1.0/4, 1.0/8, 1.0/8, 1.0/25, 1.0/60, 1.0/250, 1.0/250, 1.0/640]
# imgList = ['case5/IMG_09%d.jpg' % i for i in range(22,29)]
# exposure = [1.0, 1.0/4, 1.0/8, 1.0/15, 1.0/50, 1.0/125, 1.0/250]


# imgList = ['case1/belg%03d.jpg' % i for i in range(1,5)]
# exposure = [1.0/1000, 1.0/500, 1.0/250, 1.0/125]

imgList = ['case1/belg%03d.jpg' % i for i in range(5,10)]
exposure = [1.0/60, 1.0/30, 1.0/15, 1.0/8, 1.0/4]

#read all images into a list
img_set = []
for imgFile in imgList:
    # read one image
    img = cv2.imread(os.path.join(dataPath, imgFile)).astype(np.float32)
    img_set.append(img)

width = img_set[0].shape[0]
height = img_set[0].shape[1]

# estimate g for each channel
#lam = np.exp(-4)
lam = 100
N = 100
# to save g for each image channel
g_set = []

# to save hdr result
hdr = np.zeros(shape=(img_set[0].shape), dtype=np.float32)
local_tone = np.zeros(shape=(img_set[0].shape), dtype=np.float32)
for ich in range(3):
    Z = select_Z(img_set, N, ich)
    g = radiance_map_construction(Z, exposure, lam)
    g_set.append(g)
    print g.shape

    plt.plot(g, np.arange(256))
    plt.title('Pixel Value - Radiance Map')
    plt.xlabel('radiance')
    plt.ylabel('pixel value')
    plt.savefig('radiance_curve.png')
    """
    TODO 3
    Compute the whole radiance map

    add your code here
    """
    lnE = np.zeros(shape=(img_set[0].shape), dtype=np.float32)

    for x in range(0, width):
        for y in range(0, height):
            lnE_tmp = 0
            wei_tmp = 0
            for imgIndex in range(0, len(img_set)):  
                pix_tmp = img_set[imgIndex][x][y][ich]             
                lnE_tmp = (g[int(pix_tmp)] - np.log(exposure[imgIndex]))*weight_func(pix_tmp) + lnE_tmp
                wei_tmp = weight_func(pix_tmp) + wei_tmp
            hdr[x][y][ich] = np.exp(lnE_tmp / wei_tmp)
            if wei_tmp == 0:
                hdr[x][y][ich] = 0

global_tone = hdr/(hdr+1)

result = tone_mapping(hdr, g_set)

#uncomment these lines after you generate the final result in matrix 'img'
cv2.imshow('hdr.jpg', hdr);
cv2.imshow('global_tone.jpg', global_tone);

cv2.imshow('output', result);
#cv2.imshow('output-0.5', np.power(result, 0.5));
#cv2.imshow('output-2', np.power(result, 2));
end = time.clock()
print "Performance: time =", end - start

cv2.waitKey(0)
#cv2.imwrite('result.hdr', hdr);
cv2.imwrite('global_tone.jpg', global_tone*255.0)
cv2.imwrite('local_tone.jpg', result*255.0)






















