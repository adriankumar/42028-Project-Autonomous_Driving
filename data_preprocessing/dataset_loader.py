import os, random, h5py, torch, numpy as np
from torch.utils.data import IterableDataset, DataLoader
from data_preprocessing.video_preprocess import align_telemetry, augment_clip

#hard‑coded frame slices for training files only (start inclusive, end exclusive; -1 == to end)
TRAIN_SPLITS = {
    '2016-06-08--11-46-01.h5': (4742, 16349),
    '2016-02-08--14-56-28.h5': (3725, -1),
    '2016-06-02--21-39-29.h5': (23998, 44774),
    '2016-02-02--10-16-58.h5': (15783, 51812),
    '2016-03-29--10-50-20.h5': (14072, 76660)
    # '2016-02-11--21-32-47.h5': (989, 82285), #excluding for now, largest file causes training to last too long, may use transfer learning or move to testing/validation
}


#class extends iterable dataset and handles loading in only segmented sequences in memory during training
#yields results within loop of total_frames so that entire video sample is used, preventing discontinuity with mixed video sample inputs
#using return statement would end up loading the entire video sample in memory, use yield instead - AI suggestion
class StreamDataset(IterableDataset):
    def __init__(self, video_file, label_file, telemetry_keys,
                 seq_len=16, augment='', normalise=True,
                 start=0, end=-1, file_id=None, extra_telemetry=None):
        
        super().__init__()
        self.video_file = video_file #video file path
        self.label_file = label_file #label file path
        self.telemetry_keys = telemetry_keys #telemtry keys for labels
        self.extra_telemetry = extra_telemetry or []  #additional telemetry for features, speed_abs
        self.seq_len = seq_len #fixed seq_len
        self.augment = augment #augment type
        self.normalise = normalise
        self.start_index = start #video clipped, start at this frame for entire video sample
        self.end_index = end #end clip, end at this frame for entire video sample
        self.file_id = file_id if file_id is not None else os.path.basename(video_file)

    #overrite iterable dataset __iter__ function and 'yield' result instead of returning to prevent full data loaded in memory
    def __iter__(self):
        #open both the video file and labels
        with h5py.File(self.video_file, 'r') as vf, \
            h5py.File(self.label_file, 'r') as lf:

            vid_ds = vf['X'] #F x 3 x H x W uint8
            ptr = np.asarray(lf['cam1_ptr']) #this needs to be loaded in full memory to properly align telemetry data

            #load and align target telemetry
            aligned = {}
            for k in self.telemetry_keys:
                raw = np.asarray(lf[k])
                if k == 'steering_angle':
                    raw = raw / 10 #apply label scaling for steering angle
                aligned[k] = align_telemetry(raw, ptr)
            
            #load and align extra telemetry features, i.e speed_abs
            extra_features = {}
            for k in self.extra_telemetry:
                raw = np.asarray(lf[k])
                extra_features[k] = align_telemetry(raw, ptr)

            #variables for segmenting
            f_total = vid_ds.shape[0]
            start = max(0, self.start_index)
            end = f_total if self.end_index < 0 or self.end_index > f_total else self.end_index

            #track start index to maintain continuity for hidden states
            for s in range(start, end - self.seq_len + 1):
                e = s + self.seq_len
                clip = vid_ds[s:e]
                
                #validate clip length
                if len(clip) != self.seq_len:
                    continue
                    
                #apply augmentation to just this clip if requested
                if self.augment in ('noise', 'dim', 'light'):
                    clip = augment_clip(np.asarray(clip), self.augment)
                    
                if self.normalise:
                    clip = clip.astype(np.float32) / 255.0
                clip = torch.tensor(clip).permute(0, 2, 3, 1) #shape: [seq_len, H, W, C]
                
                #validate target sequences
                tars = []
                valid_targets = True
                for k in self.telemetry_keys:
                    t = aligned[k][s:e]
                    if len(t) != self.seq_len:
                        valid_targets = False
                        break
                    tars.append(torch.tensor(t, dtype=torch.float32).unsqueeze(-1))
                
                if not valid_targets:
                    continue
                    
                target = torch.cat(tars, dim=-1) #shape: [seq_len, n_targets]
                
                #process extra features if any
                extra_feature_tensors = []
                if self.extra_telemetry:
                    for k in self.extra_telemetry:
                        feat = extra_features[k][s:e]
                        if len(feat) != self.seq_len:
                            valid_targets = False
                            break
                        extra_feature_tensors.append(torch.tensor(feat, dtype=torch.float32).unsqueeze(-1))
                    
                    if not valid_targets:
                        continue
                
                #combine extra features if any exist
                extra_features_tensor = torch.cat(extra_feature_tensors, dim=-1) if extra_feature_tensors else None
                
                #yield with file_id and sequential position for hidden state tracking, used within the loop so that entire video file is segmented and passed to prevent mixing video samples with other files
                yield clip, target, extra_features_tensor, self.file_id, s

#class used to further process streamDataset where we loop through all n video samples for training, creatinging individual stream dataset objects for them, which loads only their iterated sequence length (segment) in memory
class ContinuousSequenceDataset(IterableDataset):
    def __init__(self, file_pairs, telemetry_keys,
                 seq_len=16, seq_stride=1, augment_prob=0.5,
                 normalise=True, shuffle_files=True,
                 slices=None, extra_telemetry=None):
        
        super().__init__()
        self.file_pairs = file_pairs #tuple from make_continuous_loader
        self.telemetry_keys = telemetry_keys #labels
        self.extra_telemetry = extra_telemetry or [] #telemetry feature names
        self.seq_len = seq_len
        self.seq_stride = seq_stride  #stride between sequences for better continuity
        self.augment_prob = augment_prob
        self.normalise = normalise
        self.shuffle_files = shuffle_files
        self.aug_opts = ['', 'noise', 'dim', 'light'] #augment options, randomly choose between no augmentation or augmentation for video input
        self.slices = slices or {} #slices to clip entire video sample for training
        
    def __iter__(self):

        #handle file arrangement
        file_order = list(range(len(self.file_pairs)))
        
        if self.shuffle_files:
            random.shuffle(file_order)
            
        #process one file at a time to maintain continuity, but all files get processed anywyas, hence we yield results within the loop to get one video file
        #it puts the video file into a stream dataset object which only loads the segment in memory
        for file_idx in file_order:
            cam, lbl = self.file_pairs[file_idx]
            fname = os.path.basename(cam)
            start, end = self.slices.get(fname, (0, -1)) #video clipping
            aug = random.choice(self.aug_opts) if random.random() < self.augment_prob else ''
            
            #unique id for this file, we will use this to manage hidden states
            file_id = f"{file_idx}_{fname}"
            
            #create dataset for this file with proper stride for continuity
            file_ds = StreamDataset(
                cam, lbl, self.telemetry_keys,
                self.seq_len, augment=aug, normalise=self.normalise,
                start=start, end=end, file_id=file_id,
                extra_telemetry=self.extra_telemetry
            )
            
            #yield all sequences from this file before moving to next
            for seq_data in file_ds:
                yield seq_data

#custom collator definition, with expected signature function __call__ implemented to handle batch loading of the video data with speed features as well
class ContinuousSequenceCollator:
    def __call__(self, batch):
        #separate the components (now with extra features)
        clips, targets, extra_features, file_ids, positions = zip(*batch)
        
        #stack the clips and targets into batches
        clip_batch = torch.stack(clips)  #shape: [batch_size, seq_len, H, W, C]
        target_batch = torch.stack(targets)  #shape: [batch_size, seq_len, n_targets]
        
        #stack extra features if present
        if extra_features[0] is not None:
            extra_features_batch = torch.stack(extra_features)  #shape: [batch_size, seq_len, n_features]
        else:
            extra_features_batch = None
        
        #keep file_ids and positions as lists
        return clip_batch, target_batch, extra_features_batch, file_ids, positions


def make_continuous_loader(root_dir, split='train',
                         telemetry_keys=None, seq_len=16,
                         seq_stride=1, batch_size=8, augment_prob=0.5,
                         normalise=True, shuffle_files=True,
                         use_test_slices=False, test_frames=1000, 
                         speed_feature=None, excluded_files=None):

    #by default we want to predict steering angle and car acceleration
    if telemetry_keys is None:
        telemetry_keys = ['steering_angle', 'car_accel']
    
    #handle speed feature specifically for input features
    extra_telemetry = speed_feature if speed_feature is not None else []

    if excluded_files is None:
        excluded_files = ['2016-02-11--21-32-47.h5']  #files to exclude from training

    cam_dir = f"{root_dir}\\{split}\\camera"
    lbl_dir = f"{root_dir}\\{split}\\labels"
    all_files = sorted(os.listdir(cam_dir))
    files = [f for f in all_files if f not in excluded_files] #filter excluded files

    #store the camera file and its corresponding telemetry labels, assuming dataset format is split\camera & split\labels
    pairs = [(os.path.join(cam_dir, f), os.path.join(lbl_dir, f)) for f in files]

    #if using quick testing
    if use_test_slices:
        test_slice_dict = {}
        for fname in files:
            original_start, original_end = TRAIN_SPLITS.get(fname, (0, -1))
            test_slice_dict[fname] = (original_start, original_start + test_frames)
        slice_dict = test_slice_dict
    
    #get full default train slices
    else:
        slice_dict = TRAIN_SPLITS if split == 'train' else {}

    #create the continuous dataset class object to load all video files with seq_len in memory at a time so that collator function handles batch properly, maintaining temporal continuity
    ds = ContinuousSequenceDataset(
        pairs, telemetry_keys, seq_len, seq_stride,
        augment_prob, normalise, shuffle_files, slices=slice_dict,
        extra_telemetry=extra_telemetry
    )
    
    #collator function to pass into dataloader `collate_fn` argumenbt
    collator = ContinuousSequenceCollator()
    
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collator
    )

#used in training utilities script to extract the hidden state stored on cpu for each video sample, so that when a segment is processed, we update the hidden state here and call it back
#to use when the next segment is loaded into memory, so that the temporal continuity within the model is maintained
class HiddenStateManager:
    def __init__(self, hidden_size, device="cpu"):
        self.hidden_states = {}  #maps file_id to (hidden_state, last_position)
        self.hidden_size = hidden_size
        self.device = device
        
    def get_hidden_states(self, file_ids, positions, batch_size):
        #retrieve appropriate hidden states for each sequence in the batch
        #if no hidden state exists or there's a continuity gap, initialise a new one
        batch_hidden_states = []
        
        for i in range(batch_size):
            file_id = file_ids[i]
            position = positions[i]
            
            if file_id in self.hidden_states:
                prev_hidden, prev_pos = self.hidden_states[file_id]
                
                #check if this is continuous from previous position
                if position == prev_pos + 1:
                    #move hidden state to the correct device
                    batch_hidden_states.append(prev_hidden.to(self.device))
                else:
                    #discontinuity, reset hidden state
                    new_hidden = torch.zeros((1, self.hidden_size), device=self.device)
                    batch_hidden_states.append(new_hidden)
            else:
                #new file, initialize hidden state
                new_hidden = torch.zeros((1, self.hidden_size), device=self.device)
                batch_hidden_states.append(new_hidden)
        
        return torch.cat(batch_hidden_states, dim=0)
    
    def update_hidden_states(self, file_ids, positions, new_hidden_states):
        #update the stored hidden states with new values after processing
        batch_size = len(file_ids)
        
        for i in range(batch_size):
            file_id = file_ids[i]
            position = positions[i]
            #keep batch dimension and store on CPU for memory efficiency
            new_hidden = new_hidden_states[i:i+1].cpu()
            
            #store the updated hidden state and position
            self.hidden_states[file_id] = (new_hidden, position + 1)
    
    def reset_hidden_state(self, file_id=None):
        #reset hidden state for a specific file or all files
        if file_id is None:
            self.hidden_states = {}
        elif file_id in self.hidden_states:
            del self.hidden_states[file_id]


def test_continuous_loader(root_dir, n_files=2, seq_len=16, batch_size=4, use_speed=True):
    #test the continuous loader with n_files from the dataset
    #shows how hidden states are managed across batches
    
    #setup loader with limited files
    cam_dir = f"{root_dir}\\train\\camera"
    lbl_dir = f"{root_dir}\\train\\labels"
    files = sorted(os.listdir(cam_dir))[:n_files]  #limit to n_files
    pairs = [(os.path.join(cam_dir, f), os.path.join(lbl_dir, f)) for f in files]
    
    slice_dict = {f: TRAIN_SPLITS.get(f, (0, -1)) for f in files}
    
    speed_feature = ['speed_abs'] if use_speed else None
    telemetry_keys = ['steering_angle', 'car_accel']
    
    ds = ContinuousSequenceDataset(
        pairs, telemetry_keys, seq_len, seq_stride=1,
        augment_prob=0, normalise=True, shuffle_files=False,
        slices=slice_dict, extra_telemetry=speed_feature
    )
    
    collator = ContinuousSequenceCollator()
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collator
    )
    
    #create hidden state manager
    hidden_manager = HiddenStateManager(hidden_size=128)
    
    print("testing continuous loader with {} files".format(n_files))
    print("batch size: {}, sequence length: {}".format(batch_size, seq_len))
    print("using speed feature: {}".format(use_speed))
    
    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= 5:  #limit to first few batches
            break
        
        #unpack batch data - properly handle speed features
        if len(batch_data) == 5:  #with speed features
            clips, targets, speed_data, file_ids, positions = batch_data
        else:
            clips, targets, file_ids, positions = batch_data
            speed_data = None
            
        #get hidden states for this batch
        batch_hidden = hidden_manager.get_hidden_states(file_ids, positions, len(file_ids))
        
        #simulate updated hidden states after processing
        new_hidden = torch.randn((len(file_ids), 128))  #mock updated hidden states
        
        #update hidden state manager
        hidden_manager.update_hidden_states(file_ids, positions, new_hidden)
        
        #show batch information
        print(f"batch {batch_idx}:")
        print(f"  clips shape: {clips.shape}")
        print(f"  targets shape: {targets.shape}")
        if speed_data is not None:
            print(f"  speed data shape: {speed_data.shape}")
        print("  file ids:", file_ids)
        print("  positions:", positions)
        print("  hidden states:", hidden_manager.hidden_states.keys())
        print()
    
    return hidden_manager


# hidden_manager = test_continuous_loader(root_dir=r"datasets\Comma_ai_dataset", seq_len=24, batch_size=4, use_speed=True)
