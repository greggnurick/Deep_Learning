from PIL import Image
import numpy as np
from scipy import ndimage
import math
import argparse
import pickle


def normalize_map(s_map):
    norm_s_map = (s_map - np.min(s_map)) / (np.max(s_map) - np.min(s_map))
    return norm_s_map


def auc_borji(s_map, gt, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()

    Sth = S[F > 0]
    num_fixations = Sth.shape[0]
    num_pixels = S.shape[0]

    r = np.random.randint(num_pixels, size=(splits, num_fixations))
    randfix = np.zeros((splits, num_fixations))

    for i in range(splits):
        randfix[i, :] = S[r[i, :]]

    aucs = []
    for i in range(splits):
        curfix = randfix[i, :]

        allthreshes = np.arange(0, np.max(np.concatenate((Sth, curfix), axis=0)), stepsize)
        allthreshes = allthreshes[::-1]

        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1.0
        fp[-1] = 1.0

        tp[1:-1] = [float(np.sum(Sth >= thresh)) / num_fixations for thresh in allthreshes]
        fp[1:-1] = [float(np.sum(curfix >= thresh)) / num_fixations for thresh in allthreshes]

        aucs.append(np.trapz(tp, fp))

    return np.mean(aucs)


def auc_shuff(s_map, gt, other_map, splits=100, stepsize=0.1):
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    oth = other_map.flatten()

    Sth = S[F > 0]
    num_fixations = Sth.shape[0]

    ind = np.flatnonzero(oth)  # find fixation locations on other images

    num_fixations_others = min(ind.shape[0], num_fixations)
    randfix = np.zeros((splits, num_fixations_others))

    for i in range(splits):
        randind = ind[np.random.permutation(ind.shape[0])]
        randfix[i, :] = S[randind[:num_fixations_others]]

    aucs = []
    for i in range(splits):
        curfix = randfix[i, :]

        allthreshes = np.arange(0, np.max(np.concatenate((Sth, curfix), axis=0)), stepsize)
        allthreshes = allthreshes[::-1]

        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1.0
        fp[-1] = 1.0

        tp[1:-1] = [float(np.sum(Sth >= thresh)) / num_fixations for thresh in allthreshes]
        fp[1:-1] = [float(np.sum(curfix >= thresh)) / num_fixations_others for thresh in allthreshes]

        aucs.append(np.trapz(tp, fp))

    return np.mean(aucs)


def cc(s_map, gt):
    # Blur and normalize
    sigma = 19
    gt = ndimage.gaussian_filter(gt, sigma)
    gt = gt - np.min(gt)
    gt = gt / np.max(gt)

    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    a = s_map_norm
    b = gt_norm
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds')
    parser.add_argument('--gts')
    args = parser.parse_args()

    preds = pickle.load(open(args.preds, 'rb'))
    gts = pickle.load(open(args.gts, 'rb'))

    cc_scores = []
    auc_borji_scores = []
    auc_shuffled_scores = []
    for i in range(len(preds)):
        gt = gts[i]['y_original']
        pred = np.reshape(preds[i], (48, 48))
        pred = Image.fromarray((pred * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]))
        pred = np.asarray(pred, dtype='float32') / 255.
        pred = ndimage.gaussian_filter(pred, sigma=2)
        cc_scores.append(cc(pred, gt))
        auc_borji_scores.append(auc_borji(pred, np.asarray(gt, dtype=np.int)))

        # Sample 10 random fixation maps for AUC Shuffled and take their union
        other = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.int)
        randind_maps = np.random.choice(len(gts), size=10, replace=False)
        for i in range(10):
            other = other | np.asarray(gts[randind_maps[i]]['y_original'], dtype=np.int)

        auc_shuffled_scores.append(auc_shuff(pred, np.asarray(gt, dtype=np.int), other))

    # CC
    print('CC: {}'.format(np.mean(cc_scores)))
    # AUC Borji
    print('AUC Borji: {}'.format(np.mean(auc_borji_scores)))
    # Shuffled AUC
    print('AUC Shuffled: {}'.format(np.mean(auc_shuffled_scores)))
