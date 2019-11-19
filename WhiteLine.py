
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
from PIL import Image
import skimage
import skimage.segmentation
from PIL import ImageOps
from sklearn.mixture import GaussianMixture
imgs = glob.glob("Y:/member/komatsu/WhiteLineRecognition/images/0006R0/*.jpg")
import chainer
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
from sklearn.preprocessing import MinMaxScaler
resnet152 = L.ResNet152Layers()

target_channels = [4,12,25,39,49,50,55,59]

target_i = 2250

fig, axs = plt.subplots(ncols=4, figsize=(20, 20))

img_org = imgs[target_i]

superpixcels = skimage.segmentation.slic(skimage.util.img_as_float(plt.imread(img_org)), 500)

img_org = Image.open(img_org)
size = img_org.size
axs[0].imshow(img_org)
axs[0].set_title('org'+str(i))

x = L.model.vision.resnet.prepare(img_org)[np.newaxis]
layers = resnet152.extract(x, layers=['conv1', 'res2', 'res3', 'res4', 'res5', 'pool5'])
feature = layers['conv1'].data.squeeze().transpose(1, 2, 0)
feature = feature[:, :, target_channels]
feature = np.mean(feature, axis=2)
feature = MinMaxScaler().fit_transform(feature)
feature *= 255
img_feature = Image.fromarray(feature)
img_feature = img_feature.resize(size)
axs[1].imshow(img_feature)
axs[1].set_title('feature mean')

img_tmp = np.array(img_feature)
for superpixcel_class in np.unique(superpixcels):
    target = np.where(superpixcels == superpixcel_class)
    feature_tmp = [img_tmp[h, w] for h, w in zip(target[0], target[1])]
    feature_mean = np.mean(feature_tmp)
    for h, w in zip(target[0], target[1]):
        img_tmp[h, w] = feature_mean
        
img_feature_mean_superpixcels = Image.fromarray(img_tmp)
axs[2].imshow(img_feature_mean_superpixcels)
axs[2].set_title('feature mean / super pixcels')

img_tmp = img_feature_mean_superpixcels.convert('L')
img_tmp = img_tmp.crop((0, size[1]//2, size[0], size[1]))
img_tmp = np.array(img_tmp)
img_tmp_for_gmm1 = img_tmp.flatten()
img_tmp_for_gmm2 = img_tmp_for_gmm1.reshape(-1, 1)
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(img_tmp_for_gmm2)
gmm = gmm.predict(img_tmp_for_gmm2).astype(np.float32)
class_a = gmm
class_b = -(gmm-1)
class_a_rgb = img_tmp_for_gmm1*class_a
class_b_rgb = img_tmp_for_gmm1*class_b
if (class_a_rgb.sum()/np.nonzero(class_a_rgb)[0].size) > (class_b_rgb.sum()/np.nonzero(class_b_rgb)[0].size):
    target_class = class_a
else:
    target_class = class_b
img_tmp = target_class.reshape(size[1]//2, size[0])
img_tmp *= 255
img_tmp = Image.fromarray(img_tmp)
overlay = Image.new(img_tmp.mode, size, 'black')
overlay.paste(img_tmp, (0, size[1]//2))
axs[3].imshow(overlay)
axs[3].set_title('overlayed')
overlay = ImageOps.grayscale(overlay)
overlay = ImageOps.colorize(overlay, black=(0, 0, 0), white=(0, 255, 255))
blended = Image.blend(img_org, overlay, 0.3)
plt.imshow(blended)
plt.show()