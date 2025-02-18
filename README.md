# Rethinking the Approach to Lightweight Multi-Branch Heterogeneous Image Fusion Frameworks: Infrared and Visible Image Fusion via the Parallel Mamba-KAN Framework (JOLT 2025)

![Image text](https://github.com/sunyichen1994/PMKFuse/blob/main/Figure/Mamba.png)


## Introduction

This is official implementation of ["Rethinking the approach to lightweight multi-branch heterogeneous image fusion frameworks: Infrared and visible image fusion via the parallel Mamba-KAN framework"](https://www.sciencedirect.com/science/article/pii/S0030399225002002?dgcid=coauthor) with Pytorch.


## Objective Evaluation and Subjective Visual Assessments Metrics
![Image text](https://github.com/sunyichen1994/PMKFuse/blob/main/Figure/F1.jpg)



## Efficiency and Effectiveness
![Image text](https://github.com/sunyichen1994/PMKFuse/blob/main/Figure/Figure2.png)


## Tips

The Trained Model is [here](https://pan.baidu.com/s/1yLiuprgQh47LRp3Oop16xQ?pwd=PMKF), and the code is: PMKF.


## Recommended Environment
 * causal-conv1d 1.1.0
 * CUDA 11.8
 * conda 4.11.0
 * mamba-ssm 1.2.0.post1
 * Python 3.7.16
 * PyTorch 2.1.1
 * timm 1.0.3
 * tqdm 4.66.4
 * pandas 2.2.2


## Citation

If you find this repository useful, please consider citing the following paper:

```
@article{sun2025JOLT,
  title={Rethinking the Approach to Lightweight Multi-Branch Heterogeneous Image Fusion Frameworks: Infrared and Visible Image Fusion via the Parallel Mamba-KAN Framework},
  author={Sun, Yichen and Dong, Mingli and Zhu, Lianqing},
  journal={Optics and Laser Technology},
  year={2025},
  volume={185},
  pages={112612},
  doi={10.1016/j.optlastec.2025.112612}
  }  
```


If you have any questions, feel free to contact me (sunyichen@emails.bjut.edu.cn)


## Acknowledgements

Parts of this code repository is based on the following works:

 * https://github.com/state-spaces/mamba
 * https://github.com/IvanDrokin/torch-conv-kan
