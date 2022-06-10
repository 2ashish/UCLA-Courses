#!/usr/bin/env bash

python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_10_iter_301000.pth ./data/Vid4/BIx4/calendar ./data/output/struct_10/calendar
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_10_iter_301000.pth ./data/Vid4/BIx4/city ./data/output/struct_10/city
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_10_iter_301000.pth ./data/Vid4/BIx4/foliage ./data/output/struct_10/foliage
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_10_iter_301000.pth ./data/Vid4/BIx4/walk ./data/output/struct_10/walk

python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_5_iter_301000.pth ./data/Vid4/BIx4/calendar ./data/output/struct_5/calendar
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_5_iter_301000.pth ./data/Vid4/BIx4/city ./data/output/struct_5/city
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_5_iter_301000.pth ./data/Vid4/BIx4/foliage ./data/output/struct_5/foliage
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/struct_5_iter_301000.pth ./data/Vid4/BIx4/walk ./data/output/struct_5/walk

python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_45_iter_301000.pth ./data/Vid4/BIx4/calendar ./data/output/unstruct_45/calendar
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_45_iter_301000.pth ./data/Vid4/BIx4/city ./data/output/unstruct_45/city
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_45_iter_301000.pth ./data/Vid4/BIx4/foliage ./data/output/unstruct_45/foliage
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_45_iter_301000.pth ./data/Vid4/BIx4/walk ./data/output/unstruct_45/walk

python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_50_iter_301000.pth ./data/Vid4/BIx4/calendar ./data/output/unstruct_50/calendar
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_50_iter_301000.pth ./data/Vid4/BIx4/city ./data/output/unstruct_50/city
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_50_iter_301000.pth ./data/Vid4/BIx4/foliage ./data/output/unstruct_50/foliage
python demo/restoration_video_demo.py ./configs/restorers/basicvsr/basicvsr_reds4.py ./data/trained_ckpt/unstruct_50_iter_301000.pth ./data/Vid4/BIx4/walk ./data/output/unstruct_50/walk

