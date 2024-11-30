# debug single sample
blender -b -P blender_render_human_ortho.py \
         -- --object_path /data/lipeng/human_scan/THuman2.1/mesh/0011/0011.obj \
         --smpl_path /data/lipeng/human_scan/THuman2.1/smplx/0011/0011.obj \
         --output_dir debug --engine CYCLES \
         --resolution 768 \
         --random_images 0
