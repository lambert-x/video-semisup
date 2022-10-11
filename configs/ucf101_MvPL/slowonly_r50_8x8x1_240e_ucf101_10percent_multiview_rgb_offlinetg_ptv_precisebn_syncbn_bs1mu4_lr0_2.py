_base_ = ['./slowonly_r50_8x8x1_240e_ucf101_10percent_multiview_rgb_offlinetg_ptv_precisebn_syncbn_bs1mu4.py']


optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0001)  # this lr 0.2 is used for 8 gpus


