# EPL: Empirical Prototype Learning for Deep Face Recognition

arXiv website is here: https://arxiv.org/abs/2405.12447v1.

Paperwithcode website is here: https://paperswithcode.com/paper/epl-empirical-prototype-learning-for-deep.

The pretrain model is updated at [Google drive](https://drive.google.com/drive/folders/1fByWagpxG2h4_kKpqKh84JDki72Z_H4P?usp=drive_link). Submit the .zip file under the onnx folder directly to the [MFR online server](http://iccv21-mfr.com/#/submit) and wait for the results.


| Method        | Network Dataset           | Mask  | Child. | Afri. | Cau.  | S-A.  | E-A.  | MR- All | IJB-C (1e-5)  | IJB-C (1e-4)  | LFW   | CFP   | Age   |
|:------:|:---------------:|:----:|:------:|:-----:|:----:|:----:|:----:|:----------:|:----:|:----:|:---:|:---:|:---:|
| EPL           | ResNet50 CASIA0.5M        | 40.92 | 33.13 | 51.50 | 65.95 | 62.32 | 31.23 | 51.92  | 83.38 | 90.13 | 99.45 | 96.46 | 94.47 |
| EPL           | ResNet50 WebFace4M        | 76.01 | 72.31 | 88.33 | 93.47 | 91.73 | 71.78 | 89.76  | 95.18 | 97.01  | 99.78 | 98.94 | 97.67 |
| EPL           | ResNet50 WebFace12M       | 82.69 | 80.10 | 92.64 | 95.89 | 94.77 | 77.84 | 93.14  | 95.99 | 97.36 | 99.80 | 99.01 | 97.93 |
| EPL           | ResNet100 WebFace12M      | 86.88 | 88.32 | 95.81 | 97.66 | 97.22 | 82.89 | 95.73  | 96.43 | 97.60 | 99.80 | 99.30 | 98.37 |

