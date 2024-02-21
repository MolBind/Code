# MolBind

## Requirements

Run the following command to create a new anaconda environment `MolBind`: 

```bash
conda env create -f environment.yml
```

## MolBind4M

Download the MolBind4M dataset from  [google drive link](https://drive.google.com/drive/folders/1ZZAypzXURoSQxUr42peHy-zzv9ruJKhU?usp=sharing), and unzip it under the `./MolBind4M/` directory

```bash
mkdir 3D-Protein && tar xvf 3DProtein.tar -C 3D-Protein/
mkdir 2D-Text && tar xvf Text2D.tar -C 2D-Text/
mkdir 3D-Text && tar xvf Text3D.tar -C 3D-Text/
mkdir 3D-2D && tar xvf 3D2D.tar -C 3D-2D/
```

## Pretrained Encoders

Download pretrained encoders  from  [google drive link](https://drive.google.com/drive/folders/1e7abVgT4_WfXIGOZDNi2Eq1aFZc0OOZX?usp=sharing), and unzip it under the `./backbone/` directory


## Pretrain

```bash
python pretrain.py  --devices '0,1,2,3' --batch_size 16 --max_epochs 100 
```






