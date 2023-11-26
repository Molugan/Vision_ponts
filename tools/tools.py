from PIL import Image
import numpy as np
import torch

def load_image(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype=np.float32)
    return torch.tensor(data)

def save_image(input_image, output_file):
    input_image = input_image.detach()
    if len(input_image.size()) == 4:
        if input_image.size(0) == 1:
            input_image = input_image.view(input_image.size(1),
                                           input_image.size(2),
                                           input_image.size(3))
        else:
            raise ValueError("save_image ne supporte que les batch ne contenant qu'une image")
    if len(input_image.size()) == 3:
        W, L, C = input_image.size()
        if C == 1:
            input_image = input_image.view(W, L)
        elif C != 3:
            raise ValueError("Format d'image non reconnue: une image doit avoir 1 ou 3 cannaux")
    input_image = torch.clamp(input_image, 0, 255)
    input_image = input_image.numpy().astype(np.uint8)
    im = Image.fromarray(input_image)
    im.save(output_file)
