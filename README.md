# Detect-AnyShadow

[Our Paper](https://arxiv.org/abs/2305.16698)

Detect-AnyShadow (DAS) is a tool for detecting shadows in videos. With very little user interaction, DAS can detect shadows throughout the whole video. It is built upon segment anything model (SAM) [「Paper」](https://arxiv.org/abs/2304.02643) and we have fine-tuned SAM to enable it to detect shadows. Then we leverage a long short-term network (LSTN) to enable the network for video shadow detection.


## Demo Tutorial


https://github.com/harrytea/Detect-AnyShadow/assets/44457717/c31fd723-3867-4106-a30c-fd477a10027d


Run `python demo_app.py` and refer to the operation of the upper demo.
```shell
# use 1 lsab block
python demo_app.py --model lstnt --ckpt_path ./lstn/checkpoints/lstnt --start_step 10000

# use 2 lsab block
python demo_app.py --model lstns --ckpt_path ./lstn/checkpoints/lstns --start_step 10000

# use 3 lsab block
python demo_app.py --model lstnb --ckpt_path ./lstn/checkpoints/lstnb --start_step 10000
```


**Note: [Gradio](https://gradio.app/) does not support drawing bounding boxes during the user interaction (for more information, please refer to [issue](https://github.com/gradio-app/gradio/issues/2316)). Therefore, we convert the connected regions that the user has marked into minimal bounding boxes in the code.**

## :tada: News

- 2023/07/12: Update the fine-tuning code of SAM. Update the training and testing code of LSTN.

- 2023/05/27: We released Detect-AnyShadow publicly, updated a simple version of the online demo, and provided test code for the [ViSha dataset](https://github.com/eraserNut/ViSha).



## :wrench: Pretrained Models

Download the original SAM ViT-B [here](https://pan.baidu.com/s/1gyRqtAy0A-jqGoQ3z1UX9w) (code: jx90), and put it into folder `'./checkpoint'`

Download the fine-tuned SAM ViT-B [here](https://pan.baidu.com/s/1pra3FUcjjTyR8vfitvh36A) (code: wons), and put it into folder `'./checkpoint/chk_sam'`


Download the [lstnt](https://pan.baidu.com/s/1XGFY07m8QQqjBkiueR32sg) (code: 0ml4), [lstns](https://pan.baidu.com/s/19AqaTJkd7JZl653VrW2VJQ) (code: obub), [lstnb](https://pan.baidu.com/s/1viuTSvI7-r5CumBqfa0tPg) (code: 2e18) and put it into folder `'./lstn/checkpoints'`



## :sunny: Get Started

### 1. Installation

```shell
conda create -n das python=3.10
pip install -r requirements.txt
```

### 2. Finetune SAM

`python sam_finetune.py`

Before running, modify the `training_path` to your own [visha](https://github.com/eraserNut/ViSha) path. Download the [sam_vit_b_01ec64.pth](https://github.com/facebookresearch/segment-anything) and put it to `./checkpoints` folder. The checkpoints will be saved in `./checkpoints/chk_sam`.

### 3. Test SAM

`python sam_test.py`

Before running, modify the `path` to your own [visha](https://github.com/eraserNut/ViSha) path. The results will be saved in `./results/sam` folder

### 4. Train LSTN

`CUDA_VISIBLE_DEVICES='0,1,2,3' python lstn/tools/train.py --amp --exp_name lstn --stage visha --model lstnb --gpu_num 4 --batch_size 16`

Before running, go to the `lstn/configs/default.py` file and modify `self.DIR_VISHA` param to your own visha data path. The checkpoints will be saved in `./checkpoints/lstn_xxxx` folder. The terminal output is as follows: 

```shell
I:0, LR:0.00002, Ref(Prev): L 0.562 IoU 15.0%, Curr1: L 0.560 IoU 12.4%, Curr2: L 0.571 IoU 11.1%, Curr3: L 0.569 IoU 11.1%, Curr4: L 0.556 IoU 13.8%
I:20, LR:0.00002, Ref(Prev): L 0.392 IoU 43.7%, Curr1: L 0.376 IoU 45.3%, Curr2: L 0.371 IoU 45.4%, Curr3: L 0.370 IoU 47.4%, Curr4: L 0.373 IoU 46.4%
I:40, LR:0.00002, Ref(Prev): L 0.248 IoU 58.9%, Curr1: L 0.263 IoU 60.5%, Curr2: L 0.254 IoU 62.0%, Curr3: L 0.253 IoU 64.2%, Curr4: L 0.251 IoU 63.6%
I:60, LR:0.00002, Ref(Prev): L 0.232 IoU 65.8%, Curr1: L 0.231 IoU 68.3%, Curr2: L 0.231 IoU 67.7%, Curr3: L 0.237 IoU 66.5%, Curr4: L 0.244 IoU 63.0%
I:80, LR:0.00002, Ref(Prev): L 0.205 IoU 69.2%, Curr1: L 0.200 IoU 65.4%, Curr2: L 0.221 IoU 61.7%, Curr3: L 0.217 IoU 63.3%, Curr4: L 0.216 IoU 63.8%
I:100, LR:0.00002, Ref(Prev): L 0.145 IoU 76.6%, Curr1: L 0.164 IoU 76.3%, Curr2: L 0.161 IoU 74.9%, Curr3: L 0.168 IoU 77.6%, Curr4: L 0.139 IoU 77.4%
...
```

**Here, we preloaded the dataset using numpy. If you are using original `jpg` or `png` format images, you will need to make slight modifications to the `lstn/dataloaders/train_datasets.py`.**


### 5. Test LSTN

`python lstn/tools/test.py --stage visha --exp_name lstn --model lstnb --datapath your_data_path --start_step 1000 --ckpt_path your_chk_path`

For test ViSha Dataset, put the testing data in `dataset`, list the data as follows, and run `python test.py`. The shadow mask will be saved in `results/lstn_xxxx/xxx`.

```shell
- test
    - images
        - video1
            - xx1.jpg
            - xx2.jpg
            - xx3.jpg
            ...
    - labels
        - video1
            - xx1.png
        - video2
            - xx2.png
```

our prediction [results](https://pan.baidu.com/s/1EFXuvbQ8wnn_KT1a1aCcTw) (code: uhaq) is here.

### 6. Inference

`python ./lstn/eval/evaluate.py --epoch=10000 --pred_path="./results/xxx"`


## :book: Citation

If you find our work useful in your research or applications, please cite our article using the following BibTex.

```
@article{wang2023detect,
  title={Detect any shadow: Segment anything for video shadow detection},
  author={Wang, Yonghui and Zhou, Wengang and Mao, Yunyao and Li, Houqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```


## :blush: Acknowledge

The code for this project brings heavily from the following repositories. We thank the authors for their great work.

[facebookresearch: segment-anything](https://github.com/facebookresearch/segment-anything)
[gaomingqi: Track-Anything](https://github.com/gaomingqi/Track-Anything/tree/master)
[z-x-yang: Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
[yoxu515: aot-benchmark](https://github.com/yoxu515/aot-benchmark)
