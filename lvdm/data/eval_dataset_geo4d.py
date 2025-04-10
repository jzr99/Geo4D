import torch
from torch.utils.data import Dataset
from dust3r.eval_metadata_geo4d import dataset_metadata
import os
from dust3r.utils.image import load_images_with_near_aspect_ratio
import glob
from dust3r.depth_eval import group_by_directory
import numpy as np
TAG_FLOAT = 202021.25
from PIL import Image
from dust3r.utils.vo_eval import load_traj, load_intrinsic

dataset_res_dict = {
    # "sintel": [448, 1024],
    "sintel": [576, 256], # original size is [1024, 436]
    "bonn": [512, 384], # original size is [640, 480]
    "kitti": [640, 192], # original size is [1242, 375]
    "scannet": [512, 384], # original size is [640, 288]
    "tum": [512, 384], # original size is [640, 480]
    "davis": [512, 320], # original [854, 480] 
    "custom": [512, 320]
    # "scannet": [640, 832],
    # "KITTI": [384, 1280],
    # "bonn": [512, 640],
    # "NYUv2": [448, 640],
}
fps = {
    "sintel": 24,
    "bonn": 24,
    "kitti": 10,
    "scannet": 24, # 30,
    "tum": 24,  # 30,
    "davis": 24,
    "custom": 24,
}
def depth_read_sintel(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_read_bonn(filename):
    # loads depth map D from png file
    # and returns it as a numpy array
    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth

def depth_read_kitti(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    img_pil = Image.open(filename)
    depth_png = np.array(img_pil, dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth

class EvalDataloader(Dataset):

    def __init__(self, dataset, seq_list=None, full_seq=False, save_dir=None,pose_eval_stride=1):
        self.data_ROOT = './'
        self.dataset = dataset
        self.metadata = dataset_metadata.get(dataset, dataset_metadata['sintel'])
        self.img_path = self.metadata['img_path']
        self.mask_path = self.metadata['mask_path']
        self.anno_path = self.metadata.get('anno_path', None)
        self.full_seq = full_seq
        # self.depth_pathes_parent_folder = self.metadata.get('depth_pathes_parent_folder', None)
        self.seq_list = seq_list
        if self.seq_list is None:
            if self.metadata.get('full_seq', False):
                self.full_seq = True
            else:
                self.seq_list = self.metadata.get('seq_list', [])
            if self.full_seq:
                self.seq_list = os.listdir(self.img_path)
                self.seq_list = [seq for seq in self.seq_list if os.path.isdir(os.path.join(self.img_path, seq))]
            self.seq_list = sorted(self.seq_list)
        self.total_seqs = len(self.seq_list)
        self.save_dir = save_dir
        self.pose_eval_stride = pose_eval_stride
        print(f"Total sequences: {self.total_seqs}")
        self.depth_read= None
        if self.dataset == 'sintel':
            depth_pathes_folder = [self.data_ROOT + f"data/sintel/training/depth/{seq}" for seq in self.seq_list]
            depth_pathes = []
            for depth_pathes_folder_i in depth_pathes_folder:
                depth_pathes += glob.glob(depth_pathes_folder_i + '/*.dpt')
            depth_pathes = sorted(depth_pathes)
            self.grouped_gt_depth = group_by_directory(depth_pathes)
            self.depth_read = depth_read_sintel
        elif self.dataset == 'bonn':
            depth_pathes_folder = [self.data_ROOT + f"data/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png" for seq in self.seq_list]
            depth_pathes = []
            for depth_pathes_folder_i in depth_pathes_folder:
                depth_pathes += glob.glob(depth_pathes_folder_i)
            depth_pathes = sorted(depth_pathes)
            self.grouped_gt_depth = group_by_directory(depth_pathes, idx=-2)
            self.depth_read = depth_read_bonn
        elif self.dataset == 'kitti':
            depth_pathes= glob.glob(self.data_ROOT + "data/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/*/*.png")
            depth_pathes = sorted(depth_pathes)
            self.grouped_gt_depth = group_by_directory(depth_pathes)
            self.depth_read = depth_read_kitti


    def __len__(self):
        return self.total_seqs
    
    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        # import pdb;pdb.set_trace()
        dir_path = self.metadata['dir_path_func'](self.img_path, seq)
        # Handle skip_condition
        skip_condition = self.metadata.get('skip_condition', None)
        # if skip_condition is not None and skip_condition(self.save_dir, seq):
        #     return self.__getitem__(self, (idx + 1))
        mask_path_seq_func = self.metadata.get('mask_path_seq_func', lambda mask_path, seq: None)
        mask_path_seq = mask_path_seq_func(self.mask_path, seq)
        filelist = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        filelist.sort()
        filelist = filelist[::self.pose_eval_stride]
        # TODO check if it is W H or H W
        imgs = load_images_with_near_aspect_ratio(
            filelist, size_wh=dataset_res_dict[self.dataset], verbose=False,
            dynamic_mask_root=mask_path_seq, crop=False,
        )
        
        gt_depth = None
        if self.depth_read is not None:
            if self.dataset == 'bonn':
                gt_pathes = self.grouped_gt_depth.get('rgbd_bonn_' + seq, None)
            else:
                gt_pathes = self.grouped_gt_depth.get(seq, None)
            # import pdb;pdb.set_trace()
            assert len(gt_pathes) == len(filelist)
            gt_depth = np.stack([self.depth_read(gt_path) for gt_path in gt_pathes], axis=0)
            gt_depth = torch.from_numpy(gt_depth).float() # [T, H, W]
        
        # import pdb;pdb.set_trace()
        img_all = []
        views = []
        for i, img in enumerate(imgs):
            img_all.append(img['img'])
            view = {
                'img': img['img'][0],
                'idx': (i,),
            }
            views.append(view)

        img_all = np.concatenate(img_all, axis=0) # [T, 3, H, W]
        img_all = torch.from_numpy(img_all).float().permute(1,0,2,3) # [3, T, H, W]
        data = {
                'video': img_all,
                'path': imgs[0]['instance'],
                'fps': fps[self.dataset],
                'caption': 'Output a video that assigns each 3D location in the world a consistent color.',
                'view': views,
                'seq': seq,
                }
        if gt_depth is not None:
            data['depth'] = gt_depth
        else:
            # data['depth'] = None
            pass

        gt_traj_file = self.metadata['gt_traj_func'](self.img_path, self.anno_path, seq)
        traj_format = self.metadata.get('traj_format', None)

        
        gt_intrinsics = None
        if self.dataset == 'sintel':
            gt_intrinsics = load_intrinsic(gt_traj_file)
        try:
            if self.dataset == 'sintel':
                gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=1)
                # gt_intrinsics = load_intrinsic(gt_traj_file=gt_traj_file)
            elif traj_format is not None:
                gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format)
            else:
                gt_traj = None
        except:
            print(f"Error in loading gt_traj for {seq}, skipping")
            gt_traj = None
        
        if gt_intrinsics is not None:
            data['intrinsics'] = gt_intrinsics
        
        if gt_traj is not None:
            data['gt_traj'] = gt_traj
        else:
            # data['gt_traj'] = None
            pass
        
        return data
            

if __name__ == "__main__":
    eval_dataset = 'kitti'
    seq_list = None
    dataloader = EvalDataloader(eval_dataset, seq_list=seq_list, full_seq=False)
    for i in range(len(dataloader)):
        sample = dataloader.__getitem__(i)
        # TODO pay attention to the trajectory format and frame id of the gt traj
        import pdb;pdb.set_trace()



