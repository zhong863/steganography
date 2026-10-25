import PIL
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torch
import numpy as np


def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")


def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0) * 2 - 1
    elif type(pil_imgs) == list:
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0) * 2 - 1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs


def add_margin(pil_img, top=0, right=0, bottom=0,
               left=0, color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)

    result.paste(pil_img, (left, top))
    return result


def image_grid(imgs, rows=1, cols=None,
               size=None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if not size is None:
        imgs = [img.resize((size, size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows * cols

    top = 20
    w, h = imgs[0].size
    delta = 0
    if len(imgs) > 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    grid = Image.new('RGB', size=(cols * w, rows * h + delta))
    for i, img in enumerate(imgs):
        if not delta == 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


# 读取一张图像 返回值大小 [1 3 512 512]
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)

    return image

def gray_code(n):
    """
    格雷码生成函数
    :param n: 格雷码位数
    :return: 格雷码列表
    """
    if n == 1:
        return ['0', '1']
    else:
        res = []
    old_gray_code = gray_code(n - 1)
    for i in range(len(old_gray_code)):
        res.append('0' + old_gray_code[i])
    for i in range(len(old_gray_code) - 1, -1, -1):
        res.append('1' + old_gray_code[i])
    return res