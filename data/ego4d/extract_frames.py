from email.mime import image
import os
import cv2
import json
import time
import numpy as np 
import av
from trim import _get_frames 

from joblib import Parallel, delayed
from tqdm import tqdm
import logging
logging.basicConfig(filename='frames.log')
logger = logging.getLogger(__name__)

# Ego4D Folders
ego4d_folder = '/media/SSD3/ego4d/dataset/data/v1'
data_folder = os.path.join(ego4d_folder, 'full_scale')
anns_folder = os.path.join(ego4d_folder, 'annotations')

# New task folder (frames we want)
output_dir = os.path.join(ego4d_folder, 'nlq_test_frames')
os.makedirs(output_dir, exist_ok=True)

# Task files (best to use the originals to extract the frames)
splits = ['test']  # Originals don't have test, others do
anns_files = '/home/mcescobar/test_nlq.json'#os.path.join(anns_folder, 'nlq_{}_unannotated.json')


# Based on their dataloader function
def extract_clip_frames(info):
    """
    This method is used to extract and save frames for all the 8 seconds
    clips. If the frames are already saved, it does nothing.
    """
    # Video info
    #breakpoint()
    video_id = info['video_uid']
    unique_id = info['clip_uid']
    start_sec = info['video_start_sec']
    end_sec = info['video_end_sec']

    clip_start_frame = int(start_sec * 30)#info['video_start_frame']
    clip_end_frame = int(end_sec * 30)#info['video_end_frame']

    # Frames info
    # pre_frame = info['pre_frame']['frame_number']
    # pnr_frame = info['pnr_frame']['frame_number']
    # post_frame = info['post_frame']['frame_number']
    # frames_list = [pre_frame, pnr_frame, post_frame]

    # # Do no repeat the ones that are already saved
    clip_save_path = os.path.join(output_dir, unique_id)
    _frames_list = [ i for i in range(clip_start_frame, clip_end_frame + 1, 1) ]
    frames_list = [ i for i in range(clip_start_frame, clip_end_frame + 1, 1) ]
    for fr in frames_list:
        if os.path.isfile(os.path.join(clip_save_path, f'{fr}.jpg')):
            _frames_list.remove(fr)
    if len(_frames_list) == 0:
        print(f'==> Skipped video {video_id}, clip {clip_id}')
        return None

    video_path = os.path.join(data_folder, video_id + '.mp4')
    print(f'Saving frames for {clip_save_path}...')
    os.makedirs(clip_save_path, exist_ok=True)

    # Extract video frames
    frames = []
    start = time.time()
    try:
        # There is one video file corrupted...
        with av.open(video_path) as container:
            for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):
                frame = frame.to_rgb().to_ndarray()
                frames.append(frame)
    except:
        logger.error(video_path)
        return None

    # Save video frames
    num_saved_frames = 0
    desired_shorter_side = 384
    for frame, frame_count in tqdm(zip(frames, frames_list)):
        original_height, original_width, _ = frame.shape
        if original_height < original_width:
            # Height is the shorter side
            new_height = desired_shorter_side
            new_width = np.round(
                original_width*(desired_shorter_side/original_height)
            ).astype(np.int32)
        elif original_height > original_width:
            # Width is the shorter side
            new_width = desired_shorter_side
            new_height = np.round(
                original_height*(desired_shorter_side/original_width)
            ).astype(np.int32)
        else:
            # Both are the same
            new_height = desired_shorter_side
            new_width = desired_shorter_side
        assert np.isclose(
            new_width/new_height,
            original_width/original_height,
            0.01
        )
        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(
            os.path.join(
                clip_save_path,
                f'{frame_count}.jpeg'
            ),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
        num_saved_frames += 1
    print(f'Time taken: {time.time() - start}; {num_saved_frames} '
        f'frames saved; {clip_save_path}')
    return None




# def format_images(im, clip_save_path, key, im_id, inst_id):
#     num_frame = im[key]['frame_number']
#     coco_output['images'].append(
#         {
#             'file_name': os.path.join(clip_save_path, f'{num_frame}.jpg'),
#             'height': im[key]['height'],
#             'width': im[key]['width'],
#             'id': im_id,
#         }
#     )

#     for bbox in im[key]['bbox']:
#         if bbox['object_type'] == 'object_of_change':
#             x = bbox['bbox']['x']
#             y = bbox['bbox']['y']
#             width = bbox['bbox']['width']
#             height = bbox['bbox']['height']
#             area = width * height
#             bounding_box = [x, y, width, height]

#             coco_output['annotations'].append(
#                 {
#                     'segmentation': [],
#                     'area': area,
#                     'iscrowd': 0,
#                     'ignore': 0,
#                     'image_id': im_id,
#                     'bbox': bounding_box,
#                     'category_id': 1,
#                     'id': inst_id
#                 }
#             )
#             inst_id += 1
#     return im_id + 1, inst_id


# output_json_dir = os.path.join(ego4d_folder, 'coco_annotations')
for split in splits:
    with open(anns_files.format(split), 'r') as f:
        frames = json.load(f)

    # image_id = 0
    # instance_id = 0
    # coco_output = {
    #     'info': INFO,
    #     'licenses': LICENSES,
    #     'categories': CATEGORIES,
    #     'images': [],
    #     'annotations': [],
    # }
    # for im in frames['clips']:
    #     video_id = im['video_uid']
    #     clip_id = im['clip_id']
    #     clip_save_path = os.path.join(video_id, clip_id)

    #     image_id, instance_id = format_images(
    #         im, clip_save_path, 'pre_frame', image_id, instance_id)
    #     image_id, instance_id = format_images(
    #         im, clip_save_path, 'pnr_frame', image_id, instance_id)
    #     image_id, instance_id = format_images(
    #         im, clip_save_path, 'post_frame', image_id, instance_id)

    # with open(f"{output_json_dir}/{split}.json", "w") as json_file:
    #     json.dump(coco_output, json_file, indent=4)
    #extract_clip_frames(frames[1])
    #Parallel(n_jobs=30)(delayed(extract_clip_frames)(im) for im in frames)
    for clip in frames:
        extract_clip_frames(clip)
    
