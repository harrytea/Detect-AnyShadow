import numpy as np
from PIL import Image
import torch
import threading


def label2colormap(label):

    m = label.astype(np.uint8)
    r, c = m.shape
    cmap = np.zeros((r, c, 3), dtype=np.uint8)
    cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3 | (m & 64) >> 1
    cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2 | (m & 128) >> 2
    cmap[:, :, 2] = (m & 4) << 5 | (m & 32) << 1
    return cmap


def one_hot_mask(mask, cls_num):
    # input:
    #   - mask: [10, 1, 465, 465]
    #   - cls_num: 10
    # return:
    #   - [10, 11, 456, 456]
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)
    # 这里假设cls_num=10，即除了背景类0，还有10类
    indices = torch.arange(0, cls_num+1, device=mask.device).view(1, -1, 1, 1)
    return (mask == indices).float()


def masked_image(image, colored_mask, mask, alpha=0.7):
    mask = np.expand_dims(mask > 0, axis=0)
    mask = np.repeat(mask, 3, axis=0)
    show_img = (image * alpha + colored_mask *
                (1 - alpha)) * mask + image * (1 - mask)
    return show_img


def save_image(image, path):
    im = Image.fromarray(np.uint8(image * 255.).transpose((1, 2, 0)))
    im.save(path)


# def _save_mask(mask, path, squeeze_idx=None):
#     if squeeze_idx is not None:
#         unsqueezed_mask = mask * 0
#         for idx in range(1, len(squeeze_idx)):
#             obj_id = squeeze_idx[idx]
#             mask_i = mask == idx
#             unsqueezed_mask += (mask_i * obj_id).astype(np.uint8)
#         mask = unsqueezed_mask
#     mask = Image.fromarray(mask).convert('P')
#     mask.putpalette(_palette)
#     mask.save(path)

def _save_mask(mask, path, squeeze_idx=None):
    # if squeeze_idx is not None:
    #     unsqueezed_mask = mask * 0
    #     for idx in range(1, len(squeeze_idx)):
    #         obj_id = squeeze_idx[idx]
    #         mask_i = mask == idx
    #         unsqueezed_mask += (mask_i * obj_id).astype(np.uint8)
    #     mask = unsqueezed_mask
    mask = Image.fromarray(mask)
    # mask.putpalette(_palette)
    mask.save(path)


def save_mask(mask_tensor, path, squeeze_idx=None):
    mask = (mask_tensor*255).cpu().numpy().astype('uint8')
    threading.Thread(target=_save_mask, args=[mask, path, squeeze_idx]).start()


def flip_tensor(tensor, dim=0):
    inv_idx = torch.arange(tensor.size(dim) - 1, -1, -1, device=tensor.device).long()
    tensor = tensor.index_select(dim, inv_idx)
    return tensor


def shuffle_obj_mask(mask):

    bs, obj_num, _, _ = mask.size()
    new_masks = []
    for idx in range(bs):
        now_mask = mask[idx]
        random_matrix = torch.eye(obj_num, device=mask.device)
        fg = random_matrix[1:][torch.randperm(obj_num - 1)]
        random_matrix = torch.cat([random_matrix[0:1], fg], dim=0)
        now_mask = torch.einsum('nm,nhw->mhw', random_matrix, now_mask)
        new_masks.append(now_mask)

    return torch.stack(new_masks, dim=0)
