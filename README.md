

## Compressive Transformer in Pytorch

It's Pytorch implementation of DeepMinds' <a href="https://arxiv.org/abs/1911.05507">Compressive Transformers</a>, an attentive sequence model which compresses past memories for long-range sequence learning.   The Compressive Transformer variant of <a href="https://arxiv.org/abs/1901.02860"> Transformer-XL</a> and obtains state-of-the-art language modelling results in the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc respectively. The code is based on <a href ="https://github.com/lucidrains/compressive-transformer-pytorch"> lucidrains' pytorch implementation </a> and  <a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL/pytorch"> NVIDIA JoC implementation </a> and maintained by Hyungon Ryu ( NVIDIA AI Technology Center).


<img src="./memory.png"></img>



## Install

```bash
$ git clone https://github.com/yhgon/cmtf.git 
$ pip install mogrifier
```


## prepare dataset enwiki8 

Install
```bash
$sh scripts/download_enwiki8.sh
```
## train 

 ```bash
 $python train.py
 ```


## Citations

```bibtex
@misc{rae2019compressive,
    title={Compressive Transformers for Long-Range Sequence Modelling},
    author={Jack W. Rae and Anna Potapenko and Siddhant M. Jayakumar and Timothy P. Lillicrap},
    year={2019},
    eprint={1911.05507},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@misc{parisotto2019stabilizing,
    title={Stabilizing Transformers for Reinforcement Learning},
    author={Emilio Parisotto and H. Francis Song and Jack W. Rae and Razvan Pascanu and Caglar Gulcehre and Siddhant M. Jayakumar and Max Jaderberg and Raphael Lopez Kaufman and Aidan Clark and Seb Noury and Matthew M. Botvinick and Nicolas Heess and Raia Hadsell},
    year={2019},
    eprint={1910.06764},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@inproceedings{rae-razavi-2020-transformers,
    title = "Do Transformers Need Deep Long-Range Memory?",
    author = "Rae, Jack  and
      Razavi, Ali",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.672"
}
```

```bibtex
@article{Shazeer2019FastTD,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam Shazeer},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1911.02150}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@misc{lan2019albert,
    title       = {ALBERT: A Lite BERT for Self-supervised Learning of Language Representations},
    author      = {Zhenzhong Lan and Mingda Chen and Sebastian Goodman and Kevin Gimpel and Piyush Sharma and Radu Soricut},
    year        = {2019},
    url         = {https://arxiv.org/abs/1909.11942}
}
```
