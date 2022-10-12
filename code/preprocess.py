import os
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm


def max_min(img):
    max_, min_ = img.max(), img.min()
    img = (img - min_) / (max_ - min_)
    return img

def float_uint8(img):
    return (img * 255).astype(np.uint8)

def l_rgb(img):
    return np.stack((img, img, img), axis=2)

def seg_rgb(img):
    color_seg = np.zeros([240, 240, 3]).astype(np.uint8)
    color_seg[img == 1] = [255, 0, 0]
    color_seg[img == 2] = [0, 255, 0]
    color_seg[img == 4] = [0, 0, 255]
    if 3 in img:
        print("Found 3!")
    return color_seg


if __name__ == '__main__':
    """
    Convert 3D BraTS images into several 2D images and corresponding labels.
    Raw 3d data locates in {path} folder.
    Converted 2D images will be put into 2D/{modality} folder with the file name {patient_id}/{slice_id}.png
    Converted 2D labels will be put into 2D/segs/ folder with the file name {patient_id}/{slice_id}.png

    Folder Structure for {data path}:
    {data path}
        |___BraTS2021_00000
              |___BraTS2021_00000_flair.nii.gz
              |___BraTS2021_00000_seg.nii.gz
              |___BraTS2021_00000_t1.nii.gz
              |___BraTS2021_00000_t1ce.nii.gz
              |___BraTS2021_00000_t2.nii.gz
        |___BraTS2021_00002
        ...
    """
    data_path = '/home/featurize/data/BraTS3D/'
    out_path = '/home/featurize/data/BraTS2D/'

    t1ce_out_path = os.path.join(out_path, 't1ce')
    t2_out_path = os.path.join(out_path, 't2')
    segs_out_path = os.path.join(out_path, 'segs')
    color_segs_out_path = os.path.join(out_path, 'colorsegs')

    os.makedirs(t1ce_out_path, exist_ok=True)
    os.makedirs(t2_out_path, exist_ok=True)
    os.makedirs(segs_out_path, exist_ok=True)
    os.makedirs(color_segs_out_path, exist_ok=True)

    patients = os.listdir(data_path)
    for patient in tqdm(patients):
        t1ce = float_uint8(max_min(np.array(nib.load(data_path + patient + '/' + patient + '_t1ce.nii.gz').dataobj)))
        t2 = float_uint8(max_min(np.array(nib.load(data_path + patient + '/' + patient + '_t2.nii.gz').dataobj)))
        seg = np.array(nib.load(data_path + patient + '/' + patient + '_seg.nii.gz').dataobj)
        assert t1ce.shape == t2.shape == seg.shape == (240, 240, 155)

        num_dim = t1ce.shape[2]

        for dim in range(num_dim):
            t1ce_slice = t1ce[:, :, dim]
            t2_slice = t2[:, :, dim]
            seg_slice = seg[:, :, dim]
            rgb_slice = seg_rgb(seg[:, :, dim])

            t1ce_slice_img = Image.fromarray(t1ce_slice)
            os.makedirs(t1ce_out_path + '/' + patient.split('_')[1], exist_ok=True)
            t1ce_slice_img.save(t1ce_out_path + '/' + patient.split('_')[1] + '/' + '%003d' % dim + '.png')

            t2_slice_img = Image.fromarray(t2_slice)
            os.makedirs(t2_out_path + '/' + patient.split('_')[1], exist_ok=True)
            t2_slice_img.save(t2_out_path + '/' + patient.split('_')[1] + '/' + '%003d' % dim + '.png')

            seg_slice_img = Image.fromarray(seg_slice)
            os.makedirs(segs_out_path + '/' + patient.split('_')[1], exist_ok=True)
            seg_slice_img.save(segs_out_path + '/' + patient.split('_')[1] + '/' + '%003d' % dim + '.png')

            rgb_slice_img = Image.fromarray(rgb_slice)
            os.makedirs(color_segs_out_path + '/' + patient.split('_')[1], exist_ok=True)
            rgb_slice_img.save(color_segs_out_path + '/' + patient.split('_')[1] + '/' + '%003d' % dim + '.png')
