import cv2 as cv
import numpy as np

def Edge(image, isgaussianBlur=1):  #bs 1 size size NCHW

    shape = image.shape
    # batch_size=shape[0]
    size = shape[2]

    robert_image = np.zeros(shape).astype(np.uint8)
    prewitt_image = np.zeros(shape).astype(np.uint8)
    sobel_image = np.zeros(shape).astype(np.uint8)
    laplacian_image = np.zeros(shape).astype(np.uint8)

    # image = np.reshape(image[i,:,:,:],(size,size))

    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gaussianBlur = cv.GaussianBlur(image, (3, 3), 0)  # 高斯滤波
    # ret, binary = cv.threshold(gaussianBlur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # 阈值处理
    if isgaussianBlur == 0:
        binary = image
    else:
        binary = gaussianBlur

    normalization = 255.0 / binary.max()
    binary = binary * normalization

    # Sobel算子
    x = cv.Sobel(binary, cv.CV_16S, 1, 0)
    y = cv.Sobel(binary, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    sobel_image[:,:,:] = Sobel / normalization

    binary = binary.astype(np.uint8)

    # 拉普拉斯算法
    dst = cv.Laplacian(binary, cv.CV_16S, ksize=3)
    Laplacian = cv.convertScaleAbs(dst)
    laplacian_image[:,:,:] = Laplacian / normalization

    # Roberts算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv.filter2D(binary, cv.CV_16S, kernelx)
    y = cv.filter2D(binary, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    robert_image[:, :, :] = Roberts / normalization

    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv.filter2D(binary, cv.CV_16S, kernelx)
    y = cv.filter2D(binary, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    prewitt_image[:, :, :] = Prewitt / normalization

    # edge = np.concatenate((Roberts, Prewitt, Sobel, Laplacian), axis=2)
    # print(edge.shape)

    return robert_image, prewitt_image, sobel_image, laplacian_image
