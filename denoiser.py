#denoiser.py

class Denoiser:
    # https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
    
    def __init__ (self):
        dst = cv.fastNlMeansDenoisingColored(img,None,3,7,21)