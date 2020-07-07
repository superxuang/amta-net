import os
import torch
from torch.utils import data
import itk
import numpy as np
import random

def create_folds(img_dir, fold_num, exclude_case, slicewise=False):
    fold_file_name = os.path.dirname(img_dir[:-1]) + '/{0:d}-fold-partition.txt'.format(fold_num)
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append(params[1])
    else:
        dataset = list(f[:] for f in os.listdir(img_dir))
        for exclude_item in exclude_case:
            if exclude_item in dataset:
                dataset.remove(exclude_item)
        case_num = len(dataset)
        fold_size = int(case_num / fold_num)
        random.shuffle(dataset)
        for fold_id in range(fold_num-1):
            folds[fold_id] = dataset[fold_id*fold_size:(fold_id+1)*fold_size]
        folds[fold_num-1] = dataset[(fold_num-1)*fold_size:]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(fold_num):
                for case_name in folds[fold_id]:
                    fold_file.write('{0:d} {1:s}\n'.format(fold_id, case_name))

    folds_slice = {}
    for fold_id in range(fold_num):
        folds_slice[fold_id] = []
        for case_name in folds[fold_id]:
            case_path = img_dir + case_name + '/'
            slice_filenames = os.listdir(case_path)
            slice_filenames.sort()
            folds_slice[fold_id] += list('{}/{}'.format(case_name,f[:]) for f in slice_filenames)
    folds = folds_slice
    
    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size
    
class Dataset(data.Dataset):
    def __init__(self, ids, dir_img, dir_pb, dir_oar, resample_size, resample_spacing, 
    min_hu, max_hu, oar_labels, adjacent, is_training, buffered_in_memory):    
        self.ids = ids
        self.dir_img = dir_img
        self.dir_pb = dir_pb
        self.dir_oar = dir_oar
        self.resample_size = resample_size
        self.resample_spacing = resample_spacing
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.oar_labels = oar_labels
        self.adjacent = adjacent
        self.is_training = is_training
        self.buffered_in_memory = buffered_in_memory
        self.ImageType = itk.Image[itk.SS, 2]
        self.LabelType = itk.Image[itk.UC, 2]
        self.buffer = {}
        self.buffer[dir_img] = {}
        self.buffer[dir_pb] = {}
        self.buffer[dir_oar] = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        transform = self.generate_transform()

        src_image = self.load_image(dir=self.dir_img, id=id, pixeltype=itk.SS)
        image = self.resample_image(
                        image=src_image, 
                        pixeltype=itk.SS, 
                        resample_size=self.resample_size, 
                        resample_spacing=self.resample_spacing, 
                        transform=transform, 
                        linear_interpolate=True, 
                        dtype=np.float32)
        image['array'] = self.normalize(image['array'])
        
        src_pb_mask = self.load_image(dir=self.dir_pb, id=id, pixeltype=itk.UC)
        pb_mask = self.resample_image(
                        image=src_pb_mask, 
                        pixeltype=itk.UC, 
                        resample_size=self.resample_size, 
                        resample_spacing=self.resample_spacing, 
                        transform=transform, 
                        linear_interpolate=False, 
                        dtype=np.int64)

        src_oar_mask = self.load_image(dir=self.dir_oar, id=id, pixeltype=itk.UC) 
        oar_mask = self.resample_image(
                        image=src_oar_mask, 
                        pixeltype=itk.UC, 
                        resample_size=self.resample_size, 
                        resample_spacing=self.resample_spacing, 
                        transform=transform, 
                        linear_interpolate=False, 
                        dtype=np.int64)
        oar_array = np.zeros_like(oar_mask['array'])
        for i in range(len(self.oar_labels)):
            label = self.oar_labels[i]
            oar_array[oar_mask['array'] == label] = i + 1
        oar_mask['array'] = oar_array

        # add adjacent slices
        if self.adjacent > 0:
            stack_shape = np.array(image['array'].shape)
            stack_shape[0] = self.adjacent * 2 + 1
            img_stack_array = np.zeros(stack_shape, dtype=np.float32)
            img_stack_array[self.adjacent,:] = image['array']
            casename = id.split('/')[0]
            ctr_img_id = int(id.split('/')[1].split('.')[0])
            for offset in range(-self.adjacent, self.adjacent+1):                
                adj_id = '{0:s}/{1:04d}.nii.gz'.format(casename, ctr_img_id+offset)
                adj_image_name = self.dir_img + adj_id
                if offset != 0 and os.path.exists(adj_image_name):
                    adj_src_image = self.load_image(dir=self.dir_img, id=adj_id, pixeltype=itk.SS)
                    adj_image = self.resample_image(
                                    image=adj_src_image, 
                                    pixeltype=itk.SS, 
                                    resample_size=self.resample_size, 
                                    resample_spacing=self.resample_spacing, 
                                    transform=transform, 
                                    linear_interpolate=True, 
                                    dtype=np.float32)
                    adj_image['array'] = self.normalize(adj_image['array'])
                    img_stack_array[self.adjacent+offset,:] = adj_image['array']
            
            image['array'] = img_stack_array

        image_tensor = torch.from_numpy(image['array'])
        pb_tensor = self.make_one_hot(torch.from_numpy(pb_mask['array']), num_classes=2)
        oar_tensor = self.make_one_hot(torch.from_numpy(oar_mask['array']), num_classes=3)

        output = {}
        output['data'] = image_tensor
        output['pb_label'] = pb_tensor
        output['oar_label'] = oar_tensor
        output['filename'] = id
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']

        return output

    def identity_transform(self):
        return itk.IdentityTransform[itk.D, 2].New()

    def generate_transform(self):
        if self.is_training:
            min_rotate = -0.05 # [rad]
            max_rotate = 0.05 # [rad]
            min_offset = -5.0 # [mm]
            max_offset = 5.0 # [mm]            
            euler_transform = itk.Euler2DTransform[itk.D].New()
            euler_parameters = euler_transform.GetParameters()
            euler_parameters = itk.OptimizerParameters[itk.D](euler_transform.GetNumberOfParameters())
            euler_parameters[0] = min_rotate + random.random() * (max_rotate - min_rotate) # rotate
            euler_parameters[1] = min_offset + random.random() * (max_offset - min_offset) # tranlate
            euler_parameters[2] = min_offset + random.random() * (max_offset - min_offset) # tranlate
            euler_transform.SetParameters(euler_parameters)
            return euler_transform
        else:
            return self.identity_transform()


    def resample_image(self, image, pixeltype, resample_size, resample_spacing, transform, linear_interpolate, dtype):
        imagetype = itk.Image[pixeltype, 2]
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        size = image.GetBufferedRegion().GetSize()
        output = {}
        output['org_size'] = np.array(size, dtype=int)
        output['org_spacing'] = np.array(spacing, dtype=float)
        output['org_origin'] = np.array(origin, dtype=float)

        new_size = (resample_size[0], resample_size[1])
        new_spacing = (resample_spacing[0], resample_spacing[1])
        new_origin = (
            origin[0]+size[0]*spacing[0]*0.5-new_size[0]*new_spacing[0]*0.5,
            origin[1]+size[1]*spacing[1]*0.5-new_size[1]*new_spacing[1]*0.5)

        output['size'] = np.array(new_size, dtype=int)
        output['spacing'] = np.array(new_spacing, dtype=float)
        output['origin'] = np.array(new_origin, dtype=float)

        resampler = itk.ResampleImageFilter[imagetype, imagetype].New()
        resampler.SetInput(image)
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(new_origin)
        resampler.SetTransform(transform)
        if linear_interpolate:
            resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imagetype, itk.D].New())
        else:
            resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imagetype, itk.D].New())
        resampler.SetDefaultPixelValue(0)
        resampler.Update()
        image = resampler.GetOutput()
        image_array = itk.GetArrayFromImage(image)
        image_array = image_array[np.newaxis, :].astype(dtype)
        output['array'] = image_array

        return output

    def read_image_file(self, filename, pixeltype):
        reader = itk.ImageFileReader[itk.Image[pixeltype,2]].New()
        reader.SetFileName(filename)
        reader.Update()
        image = reader.GetOutput()
        return image

    def load_image(self, dir, id, pixeltype):
        if id in self.buffer[dir]:
            image = self.buffer[dir][id]
        else:
            image = self.read_image_file(dir+id, pixeltype)
            if self.buffered_in_memory:
                self.buffer[dir][id] = image
        return image

    def normalize(self, x):
        factor = 1.0 / (self.max_hu - self.min_hu)
        x[x < self.min_hu] = self.min_hu
        x[x > self.max_hu] = self.max_hu
        #x = self.min_hu if x < self.min_hu else x
        #x = self.max_hu if x > self.max_hu else x
        x = (x - self.min_hu) * factor
        return x

    def make_one_hot(self, input, num_classes):
        """Convert class index tensor to one hot encoding tensor.
        Args:
            input: A tensor of shape [1, *]
            num_classes: An int of number of class
        Returns:
            A tensor of shape [num_classes, *]
        """
        shape = np.array(input.shape)
        shape[0] = num_classes
        shape = tuple(shape)
        one_hot = torch.zeros(shape)
        one_hot = one_hot.scatter_(0, input.cpu(), 1)

        return one_hot