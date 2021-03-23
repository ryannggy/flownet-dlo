import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from scipy.misc import imread, imresize

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)
        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')
            img3 = join(image_root, fprefix + "%04d"%(fnum+2) + '.png')
            img4 = join(image_root, fprefix + "%04d"%(fnum+3) + '.png')
            img5 = join(image_root, fprefix + "%04d"%(fnum+4) + '.png')
            img6 = join(image_root, fprefix + "%04d"%(fnum+5) + '.png')
            img7 = join(image_root, fprefix + "%04d"%(fnum+6) + '.png')
            img8 = join(image_root, fprefix + "%04d"%(fnum+7) + '.png')
            img9 = join(image_root, fprefix + "%04d"%(fnum+9) + '.png')
            img10 = join(image_root, fprefix + "%04d"%(fnum+9) + '.png')
            img11 = join(image_root, fprefix + "%04d"%(fnum+10) + '.png')
            img12 = join(image_root, fprefix + "%04d"%(fnum+11) + '.png')
            img13 = join(image_root, fprefix + "%04d"%(fnum+12) + '.png')
            img14 = join(image_root, fprefix + "%04d"%(fnum+13) + '.png')
            img15 = join(image_root, fprefix + "%04d"%(fnum+14) + '.png')
            img16 = join(image_root, fprefix + "%04d"%(fnum+15) + '.png')
            img17 = join(image_root, fprefix + "%04d"%(fnum+16) + '.png')
            img18 = join(image_root, fprefix + "%04d"%(fnum+17) + '.png')
            img19 = join(image_root, fprefix + "%04d"%(fnum+18) + '.png')
            img20 = join(image_root, fprefix + "%04d"%(fnum+19) + '.png')
            img21 = join(image_root, fprefix + "%04d"%(fnum+20) + '.png')
            img22 = join(image_root, fprefix + "%04d"%(fnum+21) + '.png')
            img23 = join(image_root, fprefix + "%04d"%(fnum+22) + '.png')
            img24 = join(image_root, fprefix + "%04d"%(fnum+23) + '.png')
            img25 = join(image_root, fprefix + "%04d"%(fnum+24) + '.png')
#             img26 = join(image_root, fprefix + "%04d"%(fnum+25) + '.png')
#             img27 = join(image_root, fprefix + "%04d"%(fnum+26) + '.png')
#             img28 = join(image_root, fprefix + "%04d"%(fnum+27) + '.png')
#             img29 = join(image_root, fprefix + "%04d"%(fnum+28) + '.png')
#             img30 = join(image_root, fprefix + "%04d"%(fnum+29) + '.png')
#             img31 = join(image_root, fprefix + "%04d"%(fnum+30) + '.png')
#             img32 = join(image_root, fprefix + "%04d"%(fnum+31) + '.png')
#             img33 = join(image_root, fprefix + "%04d"%(fnum+32) + '.png')
#             img34 = join(image_root, fprefix + "%04d"%(fnum+33) + '.png')
#             img35 = join(image_root, fprefix + "%04d"%(fnum+34) + '.png')
#             img36 = join(image_root, fprefix + "%04d"%(fnum+35) + '.png')
#             img37 = join(image_root, fprefix + "%04d"%(fnum+36) + '.png')
#             img38 = join(image_root, fprefix + "%04d"%(fnum+37) + '.png')
#             img39 = join(image_root, fprefix + "%04d"%(fnum+38) + '.png')
#             img40 = join(image_root, fprefix + "%04d"%(fnum+39) + '.png')
#             img41 = join(image_root, fprefix + "%04d"%(fnum+40) + '.png')
#             img42 = join(image_root, fprefix + "%04d"%(fnum+41) + '.png')
#             img43 = join(image_root, fprefix + "%04d"%(fnum+42) + '.png')
#             img44 = join(image_root, fprefix + "%04d"%(fnum+43) + '.png')
#             img45 = join(image_root, fprefix + "%04d"%(fnum+44) + '.png')
#             img46 = join(image_root, fprefix + "%04d"%(fnum+45) + '.png')
#             img47 = join(image_root, fprefix + "%04d"%(fnum+46) + '.png')
#             img48 = join(image_root, fprefix + "%04d"%(fnum+47) + '.png')
#             img49 = join(image_root, fprefix + "%04d"%(fnum+48) + '.png')
#             img50 = join(image_root, fprefix + "%04d"%(fnum+49) + '.png')
#             img51 = join(image_root, fprefix + "%04d"%(fnum+50) + '.png')
#             img52 = join(image_root, fprefix + "%04d"%(fnum+51) + '.png')
#             img53 = join(image_root, fprefix + "%04d"%(fnum+52) + '.png')
#             img54 = join(image_root, fprefix + "%04d"%(fnum+53) + '.png')
#             img55 = join(image_root, fprefix + "%04d"%(fnum+54) + '.png')
#             img56 = join(image_root, fprefix + "%04d"%(fnum+55) + '.png')
#             img57 = join(image_root, fprefix + "%04d"%(fnum+56) + '.png')
#             img58 = join(image_root, fprefix + "%04d"%(fnum+57) + '.png')
#             img59 = join(image_root, fprefix + "%04d"%(fnum+58) + '.png')
#             img60 = join(image_root, fprefix + "%04d"%(fnum+59) + '.png')
#             img61 = join(image_root, fprefix + "%04d"%(fnum+60) + '.png')
#             img62 = join(image_root, fprefix + "%04d"%(fnum+61) + '.png')
#             img63 = join(image_root, fprefix + "%04d"%(fnum+62) + '.png')
#             img64 = join(image_root, fprefix + "%04d"%(fnum+63) + '.png')
#             img65 = join(image_root, fprefix + "%04d"%(fnum+64) + '.png')
#             img66 = join(image_root, fprefix + "%04d"%(fnum+65) + '.png')
#             img67 = join(image_root, fprefix + "%04d"%(fnum+66) + '.png')
#             img68 = join(image_root, fprefix + "%04d"%(fnum+67) + '.png')
#             img69 = join(image_root, fprefix + "%04d"%(fnum+68) + '.png')
#             img70 = join(image_root, fprefix + "%04d"%(fnum+69) + '.png')
#             img71 = join(image_root, fprefix + "%04d"%(fnum+70) + '.png')
#             img72 = join(image_root, fprefix + "%04d"%(fnum+71) + '.png')
#             img73 = join(image_root, fprefix + "%04d"%(fnum+72) + '.png')
#             img74 = join(image_root, fprefix + "%04d"%(fnum+73) + '.png')
#             img75 = join(image_root, fprefix + "%04d"%(fnum+74) + '.png')
#             img76 = join(image_root, fprefix + "%04d"%(fnum+75) + '.png')
#             img77 = join(image_root, fprefix + "%04d"%(fnum+76) + '.png')
#             img78 = join(image_root, fprefix + "%04d"%(fnum+77) + '.png')
#             img79 = join(image_root, fprefix + "%04d"%(fnum+78) + '.png')
#             img80 = join(image_root, fprefix + "%04d"%(fnum+79) + '.png')
#             img81 = join(image_root, fprefix + "%04d"%(fnum+80) + '.png')
#             img82 = join(image_root, fprefix + "%04d"%(fnum+81) + '.png')
#             img83 = join(image_root, fprefix + "%04d"%(fnum+82) + '.png')
#             img84 = join(image_root, fprefix + "%04d"%(fnum+83) + '.png')
#             img85 = join(image_root, fprefix + "%04d"%(fnum+84) + '.png')
#             img86 = join(image_root, fprefix + "%04d"%(fnum+85) + '.png')
#             img87 = join(image_root, fprefix + "%04d"%(fnum+86) + '.png')
#             img88 = join(image_root, fprefix + "%04d"%(fnum+87) + '.png')
#             img89 = join(image_root, fprefix + "%04d"%(fnum+88) + '.png')
#             img90 = join(image_root, fprefix + "%04d"%(fnum+89) + '.png')
#             img91 = join(image_root, fprefix + "%04d"%(fnum+90) + '.png')
#             img92 = join(image_root, fprefix + "%04d"%(fnum+91) + '.png')
#             img93 = join(image_root, fprefix + "%04d"%(fnum+92) + '.png')
#             img94 = join(image_root, fprefix + "%04d"%(fnum+93) + '.png')
#             img95 = join(image_root, fprefix + "%04d"%(fnum+94) + '.png')
#             img96 = join(image_root, fprefix + "%04d"%(fnum+95) + '.png')
#             img97 = join(image_root, fprefix + "%04d"%(fnum+96) + '.png')
#             img98 = join(image_root, fprefix + "%04d"%(fnum+97) + '.png')
#             img99 = join(image_root, fprefix + "%04d"%(fnum+98) + '.png')
#             img100 = join(image_root, fprefix + "%04d"%(fnum+99) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(img3) or not isfile(file):
                continue

            self.image_list += [[img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, 
                                img11, img12, img13, img14, img15, img16, img17, img18, img19, img20,
                                img21, img22, img23, img24, img25, 
#                                 img26, img27, img28, img29, img30,
#                                 img31, img32, img33, img34, img35, img36, img37, img38, img39, img40,
#                                 img41, img42, img43, img44, img45, img46, img47, img48, img49, img50,
#                                 img51, img52, img53, img54, img55, img56, img57, img58, img59, img60,
#                                 img61, img62, img63, img64, img65, img66, img67, img68, img69, img70,
#                                 img71, img27, img73, img74, img75, img76, img77, img78, img79, img80,
#                                 img81, img82, img83, img84, img85, img86, img87, img88, img89, img90,
#                                 img91, img92, img93, img94, img95, img96, img97, img98, img99, img100
                                ]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])
        img4 = frame_utils.read_gen(self.image_list[index][3])
        img5 = frame_utils.read_gen(self.image_list[index][4])
        img6 = frame_utils.read_gen(self.image_list[index][5])
        img7 = frame_utils.read_gen(self.image_list[index][6])
        img8 = frame_utils.read_gen(self.image_list[index][7])
        img9 = frame_utils.read_gen(self.image_list[index][8])
        img10 = frame_utils.read_gen(self.image_list[index][9])
        img11 = frame_utils.read_gen(self.image_list[index][10])
        img12 = frame_utils.read_gen(self.image_list[index][11])
        img13 = frame_utils.read_gen(self.image_list[index][12])
        img14 = frame_utils.read_gen(self.image_list[index][13])
        img15 = frame_utils.read_gen(self.image_list[index][14])
        img16 = frame_utils.read_gen(self.image_list[index][15])
        img17 = frame_utils.read_gen(self.image_list[index][16])
        img18 = frame_utils.read_gen(self.image_list[index][17])
        img19 = frame_utils.read_gen(self.image_list[index][18])
        img20 = frame_utils.read_gen(self.image_list[index][19])
        img21 = frame_utils.read_gen(self.image_list[index][20])
        img22 = frame_utils.read_gen(self.image_list[index][21])
        img23 = frame_utils.read_gen(self.image_list[index][22])
        img24 = frame_utils.read_gen(self.image_list[index][23])
        img25 = frame_utils.read_gen(self.image_list[index][24])
#         img26 = frame_utils.read_gen(self.image_list[index][25])
#         img27 = frame_utils.read_gen(self.image_list[index][26])
#         img28 = frame_utils.read_gen(self.image_list[index][27])
#         img29 = frame_utils.read_gen(self.image_list[index][28])
#         img30 = frame_utils.read_gen(self.image_list[index][29])
#         img31 = frame_utils.read_gen(self.image_list[index][30])
#         img32 = frame_utils.read_gen(self.image_list[index][31])
#         img33 = frame_utils.read_gen(self.image_list[index][32])
#         img34 = frame_utils.read_gen(self.image_list[index][33])
#         img35 = frame_utils.read_gen(self.image_list[index][34])
#         img36 = frame_utils.read_gen(self.image_list[index][35])
#         img37 = frame_utils.read_gen(self.image_list[index][36])
#         img38 = frame_utils.read_gen(self.image_list[index][37])
#         img39 = frame_utils.read_gen(self.image_list[index][38])
#         img40 = frame_utils.read_gen(self.image_list[index][39])
#         img41 = frame_utils.read_gen(self.image_list[index][40])
#         img42 = frame_utils.read_gen(self.image_list[index][41])
#         img43 = frame_utils.read_gen(self.image_list[index][42])
#         img44 = frame_utils.read_gen(self.image_list[index][43])
#         img45 = frame_utils.read_gen(self.image_list[index][44])
#         img46 = frame_utils.read_gen(self.image_list[index][45])
#         img47 = frame_utils.read_gen(self.image_list[index][46])
#         img48 = frame_utils.read_gen(self.image_list[index][47])
#         img49 = frame_utils.read_gen(self.image_list[index][48])
#         img50 = frame_utils.read_gen(self.image_list[index][49])
#         img51 = frame_utils.read_gen(self.image_list[index][50])
#         img52 = frame_utils.read_gen(self.image_list[index][51])
#         img53 = frame_utils.read_gen(self.image_list[index][52])
#         img54 = frame_utils.read_gen(self.image_list[index][53])
#         img55 = frame_utils.read_gen(self.image_list[index][54])
#         img56 = frame_utils.read_gen(self.image_list[index][55])
#         img57 = frame_utils.read_gen(self.image_list[index][56])
#         img58 = frame_utils.read_gen(self.image_list[index][57])
#         img59 = frame_utils.read_gen(self.image_list[index][58])
#         img60 = frame_utils.read_gen(self.image_list[index][59])
#         img61 = frame_utils.read_gen(self.image_list[index][60])
#         img62 = frame_utils.read_gen(self.image_list[index][61])
#         img63 = frame_utils.read_gen(self.image_list[index][62])
#         img64 = frame_utils.read_gen(self.image_list[index][63])
#         img65 = frame_utils.read_gen(self.image_list[index][64])
#         img66 = frame_utils.read_gen(self.image_list[index][65])
#         img67 = frame_utils.read_gen(self.image_list[index][66])
#         img68 = frame_utils.read_gen(self.image_list[index][67])
#         img69 = frame_utils.read_gen(self.image_list[index][68])
#         img70 = frame_utils.read_gen(self.image_list[index][69])
#         img71 = frame_utils.read_gen(self.image_list[index][70])
#         img72 = frame_utils.read_gen(self.image_list[index][71])
#         img73 = frame_utils.read_gen(self.image_list[index][72])
#         img74 = frame_utils.read_gen(self.image_list[index][73])
#         img75 = frame_utils.read_gen(self.image_list[index][74])
#         img76 = frame_utils.read_gen(self.image_list[index][75])
#         img77 = frame_utils.read_gen(self.image_list[index][76])
#         img78 = frame_utils.read_gen(self.image_list[index][77])
#         img79 = frame_utils.read_gen(self.image_list[index][78])
#         img80 = frame_utils.read_gen(self.image_list[index][79])
#         img81 = frame_utils.read_gen(self.image_list[index][80])
#         img82 = frame_utils.read_gen(self.image_list[index][81])
#         img83 = frame_utils.read_gen(self.image_list[index][82])
#         img84 = frame_utils.read_gen(self.image_list[index][83])
#         img85 = frame_utils.read_gen(self.image_list[index][84])
#         img86 = frame_utils.read_gen(self.image_list[index][85])
#         img87 = frame_utils.read_gen(self.image_list[index][86])
#         img88 = frame_utils.read_gen(self.image_list[index][87])
#         img89 = frame_utils.read_gen(self.image_list[index][88])
#         img90 = frame_utils.read_gen(self.image_list[index][89])
#         img91 = frame_utils.read_gen(self.image_list[index][90])
#         img92 = frame_utils.read_gen(self.image_list[index][91])
#         img93 = frame_utils.read_gen(self.image_list[index][92])
#         img94 = frame_utils.read_gen(self.image_list[index][93])
#         img95 = frame_utils.read_gen(self.image_list[index][94])
#         img96 = frame_utils.read_gen(self.image_list[index][95])
#         img97 = frame_utils.read_gen(self.image_list[index][96])
#         img98 = frame_utils.read_gen(self.image_list[index][97])
#         img99 = frame_utils.read_gen(self.image_list[index][98])
#         img100 = frame_utils.read_gen(self.image_list[index][99])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, 
                img11, img12, img13, img14, img15, img16, img17, img18, img19, img20,
                img21, img22, img23, img24, img25,
#                 img26, img27, img28, img29, img30,
#                 img31, img32, img33, img34, img35, img36, img37, img38, img39, img40,
#                 img41, img42, img43, img44, img45, img46, img47, img48, img49, img50,
#                 img51, img52, img53, img54, img55, img56, img57, img58, img59, img60,
#                 img61, img62, img63, img64, img65, img66, img67, img68, img69, img70,
#                 img71, img27, img73, img74, img75, img76, img77, img78, img79, img80,
#                 img81, img82, img83, img84, img85, img86, img87, img88, img89, img90,
#                 img91, img92, img93, img94, img95, img96, img97, img98, img99, img100
                 ]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

    
class ImagesFromFolder(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    im1 = images[0]
    im2 = images[1]
    im3 = images[2]
    im4 = images[3]
    im5 = images[4]
    im6 = images[5]
    im7 = images[6]
    im8 = images[7]
    im9 = images[8]
    im10 = images[9]
    im11 = images[10]
    im12 = images[11]
    im13 = images[12]
    im14 = images[13]
    im15 = images[14]
    im16 = images[15]
    im17 = images[16]
    im18 = images[17]
    im19 = images[18]
    im20 = images[19]
    im21 = images[20]
    im22 = images[21]
    im23 = images[22]
    im24 = images[23]
    im25 = images[24]
#     im26 = images[25]
#     im27 = images[26]
#     im28 = images[27]
#     im29 = images[28]
#     im30 = images[29]
#     im31 = images[30]
#     im32 = images[31]
#     im33 = images[32]
#     im34 = images[33]
#     im35 = images[34]
#     im36 = images[35]
#     im37 = images[36]
#     im38 = images[37]
#     im39 = images[38]
#     im40 = images[39]
#     im41 = images[40]
#     im42 = images[41]
#     im43 = images[42]
#     im44 = images[43]
#     im45 = images[44]
#     im46 = images[45]
#     im47 = images[46]
#     im48 = images[47]
#     im49 = images[48]
#     im50 = images[49]
    self.image_list += [ [ im1, im2, im3, im4, im5, im6, im7, im8, im9, im10,
                         im11, im12, im13, im14, im15, im16, im17, im18, im19, im20,
                         im21, im22, im23, im24, im25,
#                          im26, im27, im28, im29, im30,
#                          im31, im32, im33, im34, im35, im36, im37, im38, im39, im40,
#                          im41, im42, im43, im44, im45, im46, im47, im48, im49, im50
                         ] ]

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0]); print(img1)
    img2 = frame_utils.read_gen(self.image_list[index][1])
    img3 = frame_utils.read_gen(self.image_list[index][2])
    img4 = frame_utils.read_gen(self.image_list[index][3])
    img5 = frame_utils.read_gen(self.image_list[index][4])
    img6 = frame_utils.read_gen(self.image_list[index][5])
    img7 = frame_utils.read_gen(self.image_list[index][6])
    img8 = frame_utils.read_gen(self.image_list[index][7])
    img9 = frame_utils.read_gen(self.image_list[index][8])
    img10 = frame_utils.read_gen(self.image_list[index][9])
    img11 = frame_utils.read_gen(self.image_list[index][0])
    img12 = frame_utils.read_gen(self.image_list[index][11])
    img13 = frame_utils.read_gen(self.image_list[index][12])
    img14 = frame_utils.read_gen(self.image_list[index][13])
    img15 = frame_utils.read_gen(self.image_list[index][14])
    img16 = frame_utils.read_gen(self.image_list[index][15])
    img17 = frame_utils.read_gen(self.image_list[index][16])
    img18 = frame_utils.read_gen(self.image_list[index][17])
    img19 = frame_utils.read_gen(self.image_list[index][18])
    img20 = frame_utils.read_gen(self.image_list[index][19])
    img21 = frame_utils.read_gen(self.image_list[index][20])
    img22 = frame_utils.read_gen(self.image_list[index][21])
    img23 = frame_utils.read_gen(self.image_list[index][22])
    img24 = frame_utils.read_gen(self.image_list[index][23])
    img25 = frame_utils.read_gen(self.image_list[index][24])
#     img26 = frame_utils.read_gen(self.image_list[index][25])
#     img27 = frame_utils.read_gen(self.image_list[index][26])
#     img28 = frame_utils.read_gen(self.image_list[index][27])
#     img29 = frame_utils.read_gen(self.image_list[index][28])
#     img30 = frame_utils.read_gen(self.image_list[index][29])
#     img31 = frame_utils.read_gen(self.image_list[index][30])
#     img32 = frame_utils.read_gen(self.image_list[index][31])
#     img33 = frame_utils.read_gen(self.image_list[index][32])
#     img34 = frame_utils.read_gen(self.image_list[index][33])
#     img35 = frame_utils.read_gen(self.image_list[index][34])
#     img36 = frame_utils.read_gen(self.image_list[index][35])
#     img37 = frame_utils.read_gen(self.image_list[index][36])
#     img38 = frame_utils.read_gen(self.image_list[index][37])
#     img39 = frame_utils.read_gen(self.image_list[index][38])
#     img40 = frame_utils.read_gen(self.image_list[index][39])
#     img41 = frame_utils.read_gen(self.image_list[index][40])
#     img42 = frame_utils.read_gen(self.image_list[index][41])
#     img43 = frame_utils.read_gen(self.image_list[index][42])
#     img44 = frame_utils.read_gen(self.image_list[index][43])
#     img45 = frame_utils.read_gen(self.image_list[index][44])
#     img46 = frame_utils.read_gen(self.image_list[index][45])
#     img47 = frame_utils.read_gen(self.image_list[index][46])
#     img48 = frame_utils.read_gen(self.image_list[index][47])
#     img49 = frame_utils.read_gen(self.image_list[index][48])
#     img50 = frame_utils.read_gen(self.image_list[index][49])
                    
    images = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, 
            img11, img12, img13, img14, img15, img16, img17, img18, img19, img20,
            img21, img22, img23, img24, img25,
#             img26, img27, img28, img29, img30,
#             img31, img32, img33, img34, img35, img36, img37, img38, img39, img40,
#             img41, img42, img43, img44, img45, img46, img47, img48, img49, img50,
#             img51, img52, img53, img54, img55, img56, img57, img58, img59, img60,
#             img61, img62, img63, img64, img65, img66, img67, img68, img69, img70,
#             img71, img27, img73, img74, img75, img76, img77, img78, img79, img80,
#             img81, img82, img83, img84, img85, img86, img87, img88, img89, img90,
#             img91, img92, img93, img94, img95, img96, img97, img98, img99, img100
             ]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

""" class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates """

class MpiSintelClean(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

class FlyingChairs(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/FlyingChairs_release/data', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    images = sorted( glob( join(root, '*.ppm') ) )

    self.flow_list = sorted( glob( join(root, '*.flo') ) )

    assert (len(images)//2 == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = images[2*i]
        im2 = images[2*i + 1]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/flyingthings3d', dstype = 'frames_cleanpass', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image_dirs = sorted(glob(join(root, dstype, 'TRAIN/*/*')))
    image_dirs = sorted([join(f, 'left') for f in image_dirs] + [join(f, 'right') for f in image_dirs])

    flow_dirs = sorted(glob(join(root, 'optical_flow_flo_format/TRAIN/*/*')))
    flow_dirs = sorted([join(f, 'into_future/left') for f in flow_dirs] + [join(f, 'into_future/right') for f in flow_dirs])

    assert (len(image_dirs) == len(flow_dirs))

    self.image_list = []
    self.flow_list = []

    for idir, fdir in zip(image_dirs, flow_dirs):
        images = sorted( glob(join(idir, '*.png')) )
        flows = sorted( glob(join(fdir, '*.flo')) )
        for i in range(len(flows)):
            self.image_list += [ [ images[i], images[i+1] ] ]
            self.flow_list += [flows[i]]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)

class ChairsSDHom(data.Dataset):
  def __init__(self, args, is_cropped, root = '/path/to/chairssdhom/data', dstype = 'train', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates

    image1 = sorted( glob( join(root, dstype, 't0/*.png') ) )
    image2 = sorted( glob( join(root, dstype, 't1/*.png') ) )
    self.flow_list = sorted( glob( join(root, dstype, 'flow/*.flo') ) )

    assert (len(image1) == len(self.flow_list))

    self.image_list = []
    for i in range(len(self.flow_list)):
        im1 = image1[i]
        im2 = image2[i]
        self.image_list += [ [ im1, im2 ] ]

    assert len(self.image_list) == len(self.flow_list)

    self.size = len(self.image_list)

    self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.image_list[index][0])
    img2 = frame_utils.read_gen(self.image_list[index][1])

    flow = frame_utils.read_gen(self.flow_list[index])
    flow = flow[::-1,:,:]

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    flow = cropper(flow)


    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)

    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class ChairsSDHomTrain(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTrain, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'train', replicates = replicates)

class ChairsSDHomTest(ChairsSDHom):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(ChairsSDHomTest, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'test', replicates = replicates)

'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''

class Tachy(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

class TachyClean(Tachy):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(TachyClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'clean', replicates = replicates)

class TachyFinal(Tachy):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(TachyFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)
