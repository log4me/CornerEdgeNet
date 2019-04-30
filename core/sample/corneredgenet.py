import cv2
import math
import numpy as np
import torch

from .utils import random_crop, draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_


def _resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
                ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections


def cornernet(system_configs, db, k_ind, data_aug, debug):
    data_rng = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories = db.configs["categories"]
    input_size = db.configs["input_size"]
    output_size = db.configs["output_sizes"][0]

    border = db.configs["border"]
    lighting = db.configs["lighting"]
    rand_crop = db.configs["rand_crop"]
    rand_color = db.configs["rand_color"]
    rand_scales = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou = db.configs["gaussian_iou"]
    gaussian_rad = db.configs["gaussian_radius"]

    # TODO. What's the meanning of max_tag_len?
    max_tag_len = 128
    # allocating memory
    images = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    # props
    gt_tl_props = np.zeros((batch_size, categories, output_size[0], output_size[1], ), dtype=np.float32)


    # offset
    gt_tl_offsets = np.zeros((batch_size, categories, 2, output_size[0], output_size[1], ), dtype=np.float32)
    # edges
    gt_tl_edges = np.zeros((batch_size, categories, 2, output_size[0], output_size[1], ), dtype=np.float32)

    gt_tl_masks = np.zeros((batch_size, categories, output_size[0], output_size[1], ), dtype=np.uint8)

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind = (k_ind + 1) % db_size

        # reading image
        image_path = db.image_path(db_ind)
        image = cv2.imread(image_path)

        # reading detections
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)

        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        width_ratio = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):
            category = int(detection[-1]) - 1

            # top left point
            xtl, ytl = detection[0], detection[1]
            # bottom right point
            xbr, ybr = detection[2], detection[3]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            if gaussian_bump:
                # bbox width, bbox height
                width = detection[2] - detection[0]
                height = detection[3] - detection[1]

                cwidth = math.ceil(width * width_ratio)
                cheight = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((cheight, cwidth), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                # TODO. Should I augment offset and width and edge as well. temporary no.
                draw_gaussian(gt_tl_props[b_ind, category], [xtl, ytl], radius)
            else:
                gt_tl_props[b_ind, category, ytl, xtl] = 1
            gt_tl_masks[b_ind, category, ytl, xtl] = 1
            offsets = [fxtl - xtl, fytl - ytl]
            gt_tl_offsets[b_ind, category, :,ytl, xtl] = offsets
            # TODO. How to represent width and height, GIOU loss or IOU Loss or SL1 Loss?
            # TODO. If use GIOU / IOU Loss, use teacher forcing or not?
            gt_tl_edges[b_ind, category, :, ytl, xtl] = [width * width_ratio - offsets[0], height * height_ratio - offsets[1]]



    images = torch.from_numpy(images)
    gt_tl_props = torch.from_numpy(gt_tl_props)
    gt_tl_offsets = torch.from_numpy(gt_tl_offsets)
    gt_tl_edges = torch.from_numpy(gt_tl_edges)
    gt_tl_masks = torch.from_numpy(gt_tl_masks)

    return {
               "xs": [images],
               "ys": [gt_tl_props, gt_tl_offsets, gt_tl_edges, gt_tl_masks]
           }, k_ind
