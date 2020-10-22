

## Compressive Transformer in Pytorch

It's community Pytorch implementation of DeepMinds' <a href="https://arxiv.org/abs/1911.05507">Compressive Transformers</a>, an attentive sequence model which compresses past memories for long-range sequence learning.   The Compressive Transformer variant of <a href="https://arxiv.org/abs/1901.02860"> Transformer-XL</a> and obtains state-of-the-art language modelling results in the WikiText-103 and Enwik8 benchmarks, achieving 17.1 ppl and 0.97 bpc respectively. The code is based on <a href ="https://github.com/lucidrains/compressive-transformer-pytorch"> lucidrains' pytorch implementation </a> and  <a href="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL/pytorch"> NVIDIA JoC implementation </a> and maintained by Hyungon Ryu ( NVIDIA AI Technology Center).


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

## configuration 
- train params 
  - optimizer : Adam
  - learning rate schedule : linear warmup from 1e-6 to 3e-4
   - for Char : 4000 warmup steps with 100,1000 decay steps
   - for Word : 16,000 warmup steps with 500,000 decay steps
   
- model params (large)
  - layers 24 
  - d_model 1024 
  - n_head 8
  - d_head 128
  - d_inner 3072  
  - seq win size : 768 
  - training memory size : 768
  - training compressive memrory size :  1152
  - compression rate : 3
  - evaluation memory size : 3072  
 
- model params (base from transformer-XL)
  - layers 12 
  - d_model 512
  - n_head 8
  - d_head 64
  - d_inner 2048
  - seq win size : 512 
  - training memory size : 512
  - training compressive memrory size :  1024
  - compression rate : 3
  - evaluation memory size : 2100 

