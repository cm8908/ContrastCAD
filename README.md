# ContrastCAD

Official python implementation for the paper: [Contrastive Learning-based Representation Learning for Computer-Aided Design Models](https://ieeexplore.ieee.org/document/10559801)

(Updated 04-30-2024) Now you can use [Docker](https://hub.docker.com/r/fmsjung/contrastcad/) to train & test our model, without installing all the dependencies locally, by running: ```docker run --gpus all -it --rm fmsjung/contrastcad```

## Training Example
You can start training ContrastCAD with desired number of epoch by running below command (-g 0 is for GPU id):
```
python train_cl.py --exp_name contrastcad -g 0
```
If you would like to use out RRE data augmentation, simply attach some arguments as below:
```
python train_cl.py --exp_name contrastcad -g 0 --augment --dataset_augment_type rre
```
Once your ContrastCAD is done training, you may train latent-GAN as latent generative model for your ContrastCAD:
```
# encode data
python test_cl.py --exp_name contrastcad --mode enc --ckpt {epoch-number-here} -g 0

# train lgan
python lgan.py --exp_name contrastcad --ae_ckpt {epoch-number-here} -g 0
```
## Evaluation Example
To evaluate the reconstruction performance:
```
# reconstruct data
python test_cl.py --exp_name contrastcad --mode rec --ckpt {epoch-number-here} -g 0

# evaluate
cd evaluation
# for accuracy
python evaluate_ae_acc.py --src ../proj_log/contrastcad/results/test_{epoch_number-here}
# for CD and invalid rate
python evaluate_ae_cd.py --src ../proj_log/contrastcad/results/test_{epoch_number-here} --parallel
```

If you would like to use our pretrained checkpoints, simply replace `exp_name` arguemtn with `pretrained` such as :
```
python test_cl.py --exp_name pretrained --mode rec --ckpt {epoch-number-here} -g 0
```
## Dataset
For dataset, please refer to [DeepCAD](https://github.com/ChrisWu1997/DeepCAD) repository. Download dataset and unzip them into `data` directory (or whatever directory name you specify).

## Citation
```
@ARTICLE{10559801,
  author={Jung, Minseop and Kim, Minseong and Kim, Jibum},
  journal={IEEE Access}, 
  title={ContrastCAD: Contrastive Learning-based Representation Learning for Computer-Aided Design Models}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Solid modeling;Shape measurement;Computational modeling;Data models;Training;Three-dimensional displays;Transformers;Design automation;Contrastive learning;CAD model;Transformer autoencoder;CAD generation},
  doi={10.1109/ACCESS.2024.3415816}}

