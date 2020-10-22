

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

 
