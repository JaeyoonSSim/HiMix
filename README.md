# [WACV'26] HiMix : Hierarchical Visual-Textual Mixing Network for Lesion Segmentation

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72086-4_48) | [Project Page](https://jaeyoonssim.github.io/publications/wacv2026/wacv2026_v1.github.io-main/index.html)

- This is the official PyTorch implementation of HiMix : Hierarchical Visual-Textual Mixing Network for Lesion Segmentation.

![overview](img/figure1.png)

## Abstract
Lesion segmentation is an essential task in medical imaging to support diagnosis and assessment of pulmonary diseases. While deep learning models have shown success in various domains, their reliance on large-scale annotated datasets limits applicability in the medical domain due to labeling cost. To address this issue, recent studies in medical image segmentation have utilized clinical texts as complementary semantic cues without additional annotations. However, most existing methods utilize a single textual embedding and fail to capture hierarchical interactions between language and visual features, which limits their ability to leverage fine-grained cues essential for precise and detailed segmentation. In this regime, we propose Hierarchical Visual-Textual Mixing Network (HiMix), a novel multi-modal segmentation framework that mixes multi-scale image and text representations throughout the mask decoding process. HiMix progressively injects hierarchical text embedding, from high-level semantics to fine-grained spatial details, into corresponding image decoder layers to bridge the modality gap and enhance visual feature refinement at multiple levels of abstraction. Experiments on the QaTa-COV19 and MosMedData+ datasets demonstrate that HiMix consistently outperforms uni-modal and multi-modal methods. Furthermore, HiMix exhibits strong generalization to unstructured textual formats, highlighting its practical applicability in real-world clinical scenarios.

## Citation
If you find our work useful for your research, please cite the our paper:
```
@inproceedings{sim2024multi,
  title={Multi-modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification},
  author={Sim, Jaeyoon and Lee, Minjae and Wu, Guorong and Kim, Won Hwa},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={511--521},
  year={2024},
  organization={Springer}
}
```

## Acknowledgements
Our work is built based on [MMI-UNet](https://github.com/nguyenpbui/MMI-UNet) and [GuideDecoder](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023). We really thank the authors for making the source code publicly available.
