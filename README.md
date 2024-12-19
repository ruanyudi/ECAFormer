# Low-light Image Enhancement using Cross Attention

## 1. Model Architecture
- Below are some examples of the model's architecture and results:
  
  ![Model Architecture](./figs/fig2.jpg)  
  ![Result 1](./figs/fig8.jpg)  
  ![Result 2](./figs/fig9.jpg)  
  ![Result 3](./figs/fig10.jpg)  
  ![Result 4](./figs/fig11.jpg)  

---

## 2. Traffic-297 Dataset
- **Download the training dataset**:  
  [Traffic-297 Dataset](https://pan.baidu.com/s/1ypPd64G_xfkV4KsrwxmJIA?pwd=amk2)  
  **Password**: `amk2`

---

## 3. Model Weights and Results
Below are the pre-trained weights and corresponding results for various datasets:  

| **Dataset**   | **Checkpoints**                                                | **Results**                                                     |
|---------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| LOL-v1        | [Download](https://pan.baidu.com/s/1kQRvbwKjbZxDfxJZOEKfrQ?pwd=6rf7) (**Password**: `6rf7`) | [Result](https://pan.baidu.com/s/1KuiNQIINtIXOwB-OMjckHg?pwd=bqrb) (**Password**: `bqrb`) |
| LOL-v2-r      | [Download](https://pan.baidu.com/s/18QrxzpD3mAOcybtQYAXlcQ?pwd=vtqj) (**Password**: `vtqj`) | [Result](https://pan.baidu.com/s/1NRzIxL8bMt2cK1bJahqNgA?pwd=eqmb) (**Password**: `eqmb`) |
| LOL-v2-s      | [Download](https://pan.baidu.com/s/1CvmJhv3epp8w_shehxXGwA?pwd=ax7q) (**Password**: `ax7q`) | [Result](https://pan.baidu.com/s/1uFWrxcNf3ru1kf15tXC8ow?pwd=5he3) (**Password**: `5he3`) |
| SID           | [Download](https://pan.baidu.com/s/1FEAw6HA4Isrz8erIb08_6A?pwd=hznl) (**Password**: `hznl`) | [Result](https://pan.baidu.com/s/1PG0nNollPaN5zvlMRlu-Jg?pwd=u4un) (**Password**: `u4un`) |
| SMID          | [Download](https://pan.baidu.com/s/15HQQbq7axbyZVJDJleeJBw?pwd=ittd) (**Password**: `ittd`) | [Result](https://pan.baidu.com/s/1IqI3vKy7dPLV2S_wGVi4eQ?pwd=h9wx) (**Password**: `h9wx`) |
| SDSD-in       | [Download](https://pan.baidu.com/s/1fPixKSvnUKqgUxww-IP-lA?pwd=71zq) (**Password**: `71zq`) | [Result](https://pan.baidu.com/s/19qry8k8KXCRlOsd-cqzcWw?pwd=e5ms) (**Password**: `e5ms`) |
| SDSD-out      | [Download](https://pan.baidu.com/s/1sK5kjSubwiGEBwn3YI0hNA?pwd=7n3e) (**Password**: `7n3e`) | [Result](https://pan.baidu.com/s/16LnLtWbGLVwX9sJo7V2Ifw?pwd=bsed) (**Password**: `bsed`) |
| Traffic-297   | [Download](https://pan.baidu.com/s/1xvA4b6Zxe20Tqro7wySgMw?pwd=mspy) (**Password**: `mspy`) | [Result](https://pan.baidu.com/s/1IGM39ycrBWVDkCbMLx0HHA?pwd=z2px) (**Password**: `z2px`) |

---

## 4. Training
- The model is trained using the [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox.

### Citation:
```bibtex
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
