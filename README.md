
<div align="center">
<h1>The Diffusion Version of SAGE</h1>


</div>

## Preparation

Follow the instructions in [w-plus-adapter](https://csxmli2016.github.io/projects/w-plus-adapter/) to train the generator and e4e encoder. Then, obtain the embeddings of the training images.

Calcultae the class embeddings of the seen categories:
```
python calculate_class_embeddings.py
```


## Training
Modify the paths in ```train_sage.sh``` to your own path.

```
sh train_sage.sh
```


## Inference

To inference on a single image:
```
python inference_sage.py --input_path <IMAGE PATH>
```




## Acknowledgement
This project is built based on the excellent [w-plus-adapter](https://github.com/tencent-ailab/IP-Adapter).

