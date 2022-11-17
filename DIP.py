import cv2
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt


class Image:
    def __init__(self, imgPath=None, image=None):
        if imgPath is not None:
            self.imgPath = imgPath
            self.image = cv2.imread(self.imgPath, 0)
        else:
            self.image = image
        self.imgSize = self.image.shape
        self.guassianNoiseSigma = None

    def CircularMask(self, radius):
        h, w = self.imgSize
        m, n = h/2 + 1, w/2 + 1
        U = np.tile(np.arange(0-m, h-m), (w, 1)).T
        V = np.tile(np.arange(0-n, w-n), (h, 1))
        dist_from_center = np.sqrt(U**2 + V**2)
        mask = dist_from_center <= radius
        return mask
        
    def norm(self, I):
        return (I  - np.min(I))/ (np.max(I) - np.min(I)) * 255

    def CLSP(self):
        h, w = self.image.shape
        p = np.zeros(self.image.shape)
        p[h//2 - 1: h//2 + 2, w//2 - 1: w//2 + 2] = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])  
        return fft.fftshift(fft.fft2(p))

    def showImage(self, cat = None):
        if cat is None:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.figure(figsize=(15,20))
            cat.append(self)
            plt.imshow(np.hstack([self.norm(x.image) for x in cat]), cmap='gray')

    def Blur(self, guassianNoiseSigma, type, **kwargs):
        Model = {
            "MotionBlur": MotionBlurModel, 
            "TurbulenceBlur": TurbulenceBlurModel,
        }
        Blur = Model[type](self.imgSize, **kwargs)
        H = Blur.degradationMatrix()
        
        # frequency domain
        F = fft.fftshift(fft.fft2(self.image))
        G = H * F

        # spatial domain
        B = np.abs(fft.ifft2(G))
        # print(np.min(B), np.max(B))
        B += 255 * np.random.normal(0, guassianNoiseSigma, B.shape)
        B = np.clip(B, 0, 255)
        bluredImage = Image(image=B)
        bluredImage.guassianNoiseSigma=guassianNoiseSigma
        return bluredImage

    def InverseFilter(self,radius=np.inf, type=None, **kwargs):
        Model = {
            "MotionBlur": MotionBlurModel, 
            "TurbulenceBlur": TurbulenceBlurModel,
        }
        Blur = Model[type](self.imgSize, **kwargs)
        return self.filtering(filter = 1 / Blur.degradationMatrix(), mask = self.CircularMask(radius))

    def WienerFilter(self, radius, K, type=None, **kwargs):
        Model = {
            "MotionBlur": MotionBlurModel, 
            "TurbulenceBlur": TurbulenceBlurModel,
        }
        Blur = Model[type](self.imgSize, **kwargs)
        H = Blur.degradationMatrix()
        return self.filtering(filter = np.abs(H)**2 / (H * (K + np.abs(H)**2)), mask = self.CircularMask(radius))
    
    def CLSFilter(self, gamma, alpha=1e2, maxInteration = 2e2, radius = 120, type=None, **kwargs):
        Model = {
            "MotionBlur": MotionBlurModel, 
            "TurbulenceBlur": TurbulenceBlurModel,
        }
        Blur = Model[type](self.imgSize, **kwargs)
        H = Blur.degradationMatrix()

        mu, sigma = 0, self.guassianNoiseSigma * 255
        h, w = self.image.shape
        eta_square = h*w*(sigma  ** 2 + mu ** 2)

        def phi(gamma):
            Fhat = np.conjugate(H) / (gamma * np.abs(self.CLSP())**2 + np.abs(H)**2) * G
            R = G - H * Fhat
            r = fft.ifft2(R)
            return np.sum(np.abs(r) ** 2)

        G = fft.fftshift(fft.fft2(self.image))

        iter_cnt = 0
        L, R = 0, gamma
        while True:
            iter_cnt += 1
            # print(iter_cnt)
            M = L + (R-L)/2
            r_square = phi(M)
            # print("L,R={},{}, eta_square={}, r_square={}".format(L, R, eta_square, r_square))

            if np.abs(r_square - eta_square) < alpha or iter_cnt >= maxInteration:
                break
            else:
                if r_square < eta_square - alpha:
                    L = M
                elif r_square > eta_square + alpha:
                    R = M
        print("gamma=[{},{}], eta_square={}, r_square={}".format(L, R, eta_square, r_square))      
        return self.filtering(filter= np.conjugate(H) / (R * np.abs(self.CLSP())**2 + np.abs(H)**2), mask=self.CircularMask(radius))

    def filtering(self, filter, mask=None):
        # frequency domain
        G = fft.fftshift(fft.fft2(self.image))
        F = G * filter
        if mask is not None:
            F = F * mask
        

        # spatial domain
        f = np.abs(fft.ifft2(F))
        deblurredImage = Image(image=f)
        return deblurredImage

class MotionBlurModel:
    def __init__(self, imgSize, T=1, a=.05, b=.05):
        self.imgSize = imgSize
        self.T = T
        self.a = a
        self.b = b

    def degradationMatrix(self):
        h, w = self.imgSize
        m, n = h/2 + 1, w/2 + 1
        U = np.tile(np.arange(0-m, h-m), (w, 1)).T
        V = np.tile(np.arange(0-n, w-n), (h, 1))
        t = U * self.a + V * self.b
        H = self.T * np.sinc(t) * np.exp(-1j * t * np.pi)
        return H

class TurbulenceBlurModel:
    def __init__(self, imgSize, k = .0025):
        self.imgSize = imgSize 
        self.k = k
    
    def degradationMatrix(self):
        h, w = self.imgSize
        m, n = h/2 + 1, w/2 + 1
        U = np.tile(np.arange(0-m, h-m), (w, 1)).T
        V = np.tile(np.arange(0-n, w-n), (h, 1))
        H = np.exp(-self.k * (U**2 + V**2) ** (5/6))
        return H