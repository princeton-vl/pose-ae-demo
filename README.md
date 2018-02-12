# Associative Embedding: End-to-End Learning for Joint Detection and Grouping

This repository includes Tensorflow code for running the multi-person pose estimiation algorithm presented in:

Alejandro Newell, Zhiao Huang, and Jia Deng, 
**Associative Embedding: End-to-End Learning for Joint Detection and Grouping**, 
[arXiv:1611.05424](https://arxiv.org/abs/1611.05424v2), 2016.

Pretrained models are available [here](https://umich.box.com/s/ic7953g3dcpf9snuuw5yhx67vx88zfrp). Include the models in the main directory of this repository to run the demo code.

**Update (12/15):** PyTorch training code is now available: https://github.com/umich-vl/pose-ae-train

To run the code, the following must be installed:

- Python3
- tensorflow
- numpy
- Opencv3.0
- cudnn
- munkres
- tqdm
- scipy

## Testing your own images
To test a single image: ```python main.py --input_image_path inp.jpg --output_image_path out.jpg```

For better results (but slower evaluation), you can pass ```--scales multi``` to enable multi-scale evaluation and/or ```-r refinement``` to enable an additional refinement step. 

The prediction is visulized in ```out.jpg```

To test a set of images, put your image paths in a single file, one image a line and run ```python main.py -l image_path_list.txt -f output.json```

``output.json'' is a list of prediction per image. The data format is:

```
[{
     'image_path': str,
     'score': float,
     'keypoints': [x1,y1,s1,...,xk,yk,sk],
}]
```

Our data format is similiar to [MS-COCO](http://mscoco.org/dataset/#format). Note that ```s_i``` is the confidence score of each joint instead of visibility.
