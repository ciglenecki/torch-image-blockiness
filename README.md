# 🧱 Torch JPEG Blockiness Metric

The higher the blockiness metric value, the more likely it is that the image was JPEG-compressed at a low quality.


![](https://raw.githubusercontent.com/ciglenecki/torch-jpeg-blockiness/refs/heads/main/assets/readme.webp)


This is a implementation of blockiness algorithm from the paper ["A JPEG blocking artifact detector for image forensics" (Dinesh Bhardwaj, Vinod Pankajakshan)](https://www.sciencedirect.com/science/article/abs/pii/S0923596518302066).

It is based on the gohtanii's implementation from [DiverSeg dataset: "Rethinking Image Super-Resolution from Training Data Perspectives" (Go Ohtani, Ryu Tadokoro, Ryosuke Yamada, Yuki M. Asano, Iro Laina, Christian Rupprecht, Nakamasa Inoue, Rio Yokota, Hirokatsu Kataoka, and Yoshimitsu Aoki, ECCV2024)](https://github.com/gohtanii/DiverSeg-dataset/tree/284cc1c030424b8b0f7040020bd6435e8ed2e6d7).


**It has the following improvements over the gohtanii's implementation:**

1. 🔥 operations are written in torch (gpu friendly)
2. 🔥 operations are vectorized
3. 🔥 batched input is supported, with the assumption of same image size



## Usage

(option a) you can copy paste the [torch_jpeg_blockiness/blockiness.py](torch_jpeg_blockiness/blockiness.py) file to your project directory as it has no dependencies except torch and numpy.

(option b) you can also install the package with pip
```
pip install torch_jpeg_blockiness
```


usage:
```py
import torchvision
import torchvision.transforms.functional
from torch_jpeg_blockiness.blockiness import calculate_image_blockiness, rgb_to_grayscale

img = torchvision.io.read_image("example_images/unsplash.jpg")
img_gray = rgb_to_grayscale(img)
blockiness = calculate_image_blockiness(img_gray)
blockiness_float = float(blockiness)
```

test the code against original (gohtanii's implementation)

```bash
python3 -m test.test
```

```bash
.
------------------------
Ran 1 test in 11.771s

OK
```


## Formal definitions

Definisions from the [DiverSeg dataset: "Rethinking Image Super-Resolution from Training Data Perspectives" (Go Ohtani, Ryu Tadokoro, Ryosuke Yamada, Yuki M. Asano, Iro Laina, Christian Rupprecht, Nakamasa Inoue, Rio Yokota, Hirokatsu Kataoka, and Yoshimitsu Aoki, ECCV2024)](https://github.com/gohtanii/DiverSeg-dataset/tree/284cc1c030424b8b0f7040020bd6435e8ed2e6d7) paper:

> These definitions are missing some crucial parts that would make them easier to understand. I therefore recommend reading the original paper ["A JPEG blocking artifact detector for image forensics"](https://www.sciencedirect.com/science/article/abs/pii/S0923596518302066) with sci-hub.


![](https://raw.githubusercontent.com/ciglenecki/torch-jpeg-blockiness/refs/heads/main/assets/2025-02-27_04-01.png)
![](https://raw.githubusercontent.com/ciglenecki/torch-jpeg-blockiness/refs/heads/main/assets/2025-02-27_04-01_1.png)
![](https://raw.githubusercontent.com/ciglenecki/torch-jpeg-blockiness/refs/heads/main/assets/2025-02-27_04-02.png)


## Other

**Motivation**: Authors of the paper have demonstrated that filtering images which have JPEG compresson helps the super resolution model achieve better performance in the long run. Even more crucial, JPEG blockiness filtering prevents the dataset from containing excessively strong compression, which can be **highly detrimental to the overall training process** as shown by the [DiverSeg dataset paper](https://arxiv.org/abs/2409.00768) and [Phillip Hoffman's BHI filtering blog post](https://huggingface.co/blog/Phips/bhi-filtering#blockiness).



> note: the code is tested against the original implementation. test can be found at [tests/test.py](tests/test.py)

> note 2: this method should be shift and scale-invariant as it computes the DCT for each fixed block rather than the entire image, but I haven't empirically tested this yet.

> note 3: I was thinking about using the centarl crop of an image (most likely high entropy area) to reduce the compute time even more. The problem is that the crop might start in the middle of the JPEG block. I'm not sure if this method is robust to JPEG grid offsets. However, you could start cropping from the upper-left corner to some fixed dimension. Preferably, the bottom-right corner should pass through the center of the image.


## References

Phillip Hoffman's "Filtering single image super-resolution datasets with BHI" blog post https://huggingface.co/blog/Phips/bhi-filtering

--- 
DiverSeg dataset: "Rethinking Image Super-Resolution from Training Data Perspectives" (Ohtani, Go and Tadokoro, Ryu and Yamada, Ryosuke and Asano, Yuki M and Laina, Iro and Rupprecht, Christian and Inoue, Nakamasa and Yokota, Rio and Kataoka, Hirokatsu and Aoki, Yoshimitsu)

github: https://github.com/gohtanii/DiverSeg-dataset/tree/284cc1c030424b8b0f7040020bd6435e8ed2e6d7

arxiv: https://arxiv.org/abs/2409.00768

```
@inproceedings{ohtani2024rethinking,
  title={Rethinking Image Super-Resolution from Training Data Perspectives},
  author={Ohtani, Go and Tadokoro, Ryu and Yamada, Ryosuke and Asano, Yuki M and Laina, Iro and Rupprecht, Christian and Inoue, Nakamasa and Yokota, Rio and Kataoka, Hirokatsu and Aoki, Yoshimitsu},
  booktitle={European Conference on Computer Vision},
  pages={19--36},
  year={2024},
  organization={Springer}
}
```

--- 

"A JPEG blocking artifact detector for image forensics", original paper which defined this concrete blockiness metric.

(open with sci-hub) https://www.sciencedirect.com/science/article/abs/pii/S0923596518302066 

```
@article{bhardwaj2018jpeg,
  title={A JPEG blocking artifact detector for image forensics},
  author={Bhardwaj, Dinesh and Pankajakshan, Vinod},
  journal={Signal Processing: Image Communication},
  volume={68},
  pages={155--161},
  year={2018},
  publisher={Elsevier}
}
```
