# Detect-AnyShadow

[Paper Link](https://arxiv.org/abs/2305.16698)

Detect-AnyShadow is a tool for detecting shadows in videos. With very little user assistance, DAS can detect shadows throughout the whole video. It is built upon segment anything [(Paper)](https://arxiv.org/abs/2304.02643) and we have fine-tuned SAM to enable it to detect shadows.


## Demo Tutorial


https://github.com/harrytea/Detect-AnyShadow/assets/44457717/c31fd723-3867-4106-a30c-fd477a10027d



## :tada: News

- 2023/05/27: We released Detect-AnyShadow publicly, updated a simple version of the online demo, and provided test code for the [ViSha dataset](https://github.com/eraserNut/ViSha).


## :wrench: Pretrained Models

Download the finetuned SAM ViT-B [here](https://pan.baidu.com/s/11qqlKX_iU-bFqRDlfCtE6w) (code: ada2), and put it into folder `'./checkpoint'`

Download the LSTN [here](https://pan.baidu.com/s/1oRAnWCNG2Cy1Mk6I1uVTaA) (code: ada2), and put it into folder `'./lstn/checkpoints'`

## :sunny: Get Started

Run `python demo_app.py` and refer to the operation of the upper demo.

**Note: [Gradio](https://gradio.app/) does not support drawing bounding boxes during the user interaction (for more information, please refer to [issue](https://github.com/gradio-app/gradio/issues/2316)). Therefore, we convert the connected regions that the user has marked into minimal bounding boxes in the code.**

For test ViSha Dataset, put the testing data in `dataset`, list the data as follows, and run `python test.py`. The shadow mask will be saved in `results`.

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

## :book: Citation

If you find our work useful in your research or applications, please cite our article using the following BibTex.

```
@misc{wang2023detect,
      title={Detect Any Shadow: Segment Anything for Video Shadow Detection}, 
      author={Yonghui Wang and Wengang Zhou and Yunyao Mao and Houqiang Li},
      year={2023},
      eprint={2305.16698},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## :blush: Acknowledge

The code for this project brings heavily from the following repositories. We thank the authors for their great work.

[facebookresearch: segment-anything](https://github.com/facebookresearch/segment-anything)
[gaomingqi: Track-Anything](https://github.com/gaomingqi/Track-Anything/tree/master)
[z-x-yang: Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
[yoxu515: aot-benchmark](https://github.com/yoxu515/aot-benchmark)
