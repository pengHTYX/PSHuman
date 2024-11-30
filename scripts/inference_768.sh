seed=600
gpu=$1
CUDA_VISIBLE_DEVICES=$gpu python inference.py --config configs/inference-768-6view.yaml \
    pretrained_model_name_or_path='pengHTYX/PSHuman_Unclip_768_6views' \
    validation_dataset.crop_size=740 \
    with_smpl=false \
    validation_dataset.root_dir='examples' \
    seed=$seed \
    num_views=7 \
    save_mode='rgb' # if save rgba images for each view, if not, noly save concatenated images for visualization 



