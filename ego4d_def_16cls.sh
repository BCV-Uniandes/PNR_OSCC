# Without this export it does not work, can't import some strange library
export LD_LIBRARY_PATH=~/anaconda3/lib
export PYTHONWARNINGS="ignore"

# TRAIN THE MODEL
## 1. use_checkpoint makes the training a bit slower (don't know how much), but 
## it is much lighter and I can increase the batch size a lot, so it's faster.
## 2. If starting from a backbone pretrained on ImageNet, I need to use
## model.backbone.pretrained. If it is from a model trained on Kinetics is
## load_from
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist_train.sh \
#configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py 4 \
#--resume-from work_dirs/ego4d_swin_16cls_multigpu_weighted/latest.pth \
#--cfg-options model.backbone.use_checkpoint=True --gpu-ids 4 5 6 7  \
#model.backbone.pretrained=pretrained/swin_base_patch244_window877_kinetics600_22k.pth \
# model.backbone.num_points=4 work_dir=work_dirs/Dswin_4p_s4_attnpathdrop0 \
# data.videos_per_gpu=60 data.train.pipeline.0.frame_interval=4 \
#model.backbone.drop_path_rate=0.0
#--resume-from work_dirs/ego4d_swin_32cls_multigpu_k600/latest.pth \
#--resume-from work_dirs/ego4d_swin_16cls_multigpu_k600/latest.pth \

# # # If resume, put this BEFORE --cfg-options
# # --resume-from work_dirs/Dswin_k600_4p_lrstep/latest.pth \

# # TEST 5 
#bash tools/dist_test.sh \
# CUDA_VISIBLE_DEVICES=3 python tools/test.py \
# configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py \
# work_dirs/ego4d_swin_16cls_multigpu_weighted/latest.pth --out 'TEST_weights_scratch.json' \
#--cfg-options data.test.pipeline.0.frame_interval=4

# # TEST 5 
#bash tools/dist_test.sh 1\
#CUDA_VISIBLE_DEVICES=4 python tools/test.py \
#configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py \
#best_models/VSwinT_best.pth --eval 'top_k_accuracy' 'mean_class_accuracy' 'keyframe_distance' --out 'VAL_ego4d_PNR_best.json' \


# # TEST 5 
#bash tools/dist_test.sh \
# CUDA_VISIBLE_DEVICES=5 python tools/test.py \
# configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py \
# work_dirs/ego4d_swin_16cls_multigpu/epoch_20.pth --eval 'top_k_accuracy' 'mean_class_accuracy' 'keyframe_distance' --out 'ego4d_swin_16cls_multigpu_E20.json' \
#--cfg-options data.test.pipeline.0.frame_interval=4

# # TEST 5 
#bash tools/dist_test.sh \
# CUDA_VISIBLE_DEVICES=6 python tools/test.py \
# configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py \
# work_dirs/ego4d_swin_16cls_multigpu/epoch_25.pth --eval 'top_k_accuracy' 'mean_class_accuracy' 'keyframe_distance' --out 'ego4d_swin_16cls_multigpu_E25.json' \
#--cfg-options data.test.pipeline.0.frame_interval=4

# # Single GPU train
#CUDA_VISIBLE_DEVICES=1 python tools/train.py \
#configs/recognition/swin/swin_base_patch244_window877_ego4d_22k.py \
#--cfg-options model.backbone.use_checkpoint=True \
#load_from=work_dirs/ucf101_swin_base_k600_22k_patch244_window877/latest.pth

#CUDA_VISIBLE_DEVICES=1 python tools/train.py \
#configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_K600.py \
#--cfg-options \ #model.backbone.use_checkpoint=True \
#model.backbone.pretrained=pretrained/swin_base_patch244_window877_kinetics600_22k.pth \
# model.backbone.num_points=4 work_dir=work_dirs/Dswin_4p_s4_attnpathdrop0 \
# data.videos_per_gpu=60 data.train.pipeline.0.frame_interval=4 \
#model.backbone.drop_path_rate=0.0

# # Single GPU test (Using the python tools/test.py creates a deadlock)
# CUDA_VISIBLE_DEVICES=2 bash tools/dist_test.sh \
# configs/recognition/swin/swin_base_patch244_window877_ucf101_22k_FreeAT.py \
# work_dirs/FreeAT/base_k600_lr5_eps24/latest.pth 1 --eval top_k_accuracy

CUDA_VISIBLE_DEVICES=2 bash tools/dist_test.sh \
configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_16cls.py \
best_models/VSwinT_best.pth 1 --eval 'top_k_accuracy' 'mean_class_accuracy' 'keyframe_distance' --out 'VAL_ego4d_PNR_best.json' \
