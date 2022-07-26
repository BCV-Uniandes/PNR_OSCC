# Without this export it does not work, can't import some strange library
export LD_LIBRARY_PATH=~/anaconda3/lib
export PYTHONWARNINGS="ignore"

# TRAIN THE MODEL
## 1. use_checkpoint makes the training a bit slower (don't know how much), but 
## it is much lighter and I can increase the batch size a lot, so it's faster.
## 2. If starting from a backbone pretrained on ImageNet, I need to use
## model.backbone.pretrained. If it is from a model trained on Kinetics is
## load_from 
#bash tools/dist_train.sh \
#configs/recognition/swin/Dswin_base_patch244_window877_ego4d_k600_FreeAT.py 8 \
#--cfg-options model.backbone.use_checkpoint=True \
#model.backbone.pretrained=pretrained/DSwin_best.pth \
# model.backbone.num_points=4 work_dir=work_dirs/Dswin_4p_s4_attnpathdrop0 \
# data.videos_per_gpu=60 data.train.pipeline.0.frame_interval=4 \
#model.backbone.drop_path_rate=0.0

CUDA_VISIBLE_DEVICES=1 python tools/test.py \
configs/recognition/swin/Dswin_base_patch244_window877_ego4d_k600_FreeAT.py \
work_dirs/Dswin_bestOSSC_deltaloss/latest.pth --out 'TEST_Dswin_delta.json' \

#--resume-from work_dirs/ego4d_swin_16cls_multigpu_k600/latest.pth \

# # # If resume, put this BEFORE --cfg-options
# # --resume-from work_dirs/Dswin_k600_4p_lrstep/latest.pth \

# # TEST 5
#bash tools/dist_test.sh \
# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
# configs/recognition/swin/swin_base_patch244_window877_ego4d_22k_K600_16cls.py \
# work_dirs/ego4d_swin_16cls_multigpu_k600_lre5_weighted/latest.pth --out 'TEST_weights_pretrained_K600.json' \
#--cfg-options data.test.pipeline.0.frame_interval=4
