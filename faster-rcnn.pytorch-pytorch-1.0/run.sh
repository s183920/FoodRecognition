python trainval_net.py --dataset pascal_voc --net res101 \
                       --bs 24 --nw 8 \
                       --lr 0.0001 --lr_decay_step 1000 \
                       --cuda 


# python demo.py --net res101 \
#                --checksession 1 --checkepoch 6 --checkpoint 416 \
#                --cuda --load_dir data/pretrained_model