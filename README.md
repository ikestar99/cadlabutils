## Cadwell Lab Patch-seq Morphology Reconstruction Pipeline
Ike Ogbonna (computational), Kevin Lee, PhD (Experimental), Cathryn Cadwell, MD, PhD (Principal Investigator)

### Background
This repository incorporates model state dicts from [patchseq_autorecon](https://www.nature.com/articles/s41467-024-50728-9), which presents two convolutional neural networks trained to segment neurite and soma brightfield image stacks of biocytin-filled cortical neurons.

### Expected Input Format
Codebase expects a brightfield image stack as input. Each stack should adhere to the following:
- Stored as .tif format
- Z slices stored as individual images named in naturally ascending order, ex 000.tif, 001.tif, ..., 199.tif
- All individual images are 2D with a standardized width and height
- All images share the same data type, 8-bit unsigned integer

Using hyperstack formats as input is discouraged as it negates the main benefit of this repository -- flexible computing. Hyperstack image files  (ex. multipage .tifs) are significantly larger than individual image files, requiring much more system memory to load and manipulate. With individual tifs, the pipeline avoids loading all images into memory at one time and thus manages to process stacks significantly larger than available RAM on a computer with finite compute power.
 
### Hardware
This repository was developed on the following hardware:
- Apple M1 Pro Macbook Pro, 32gb unified memory (code design, review, unit tests)
- PC with AMD Ryzen 9 9950x, 64gb DDR5 RAM (4x 16gb DIMMs), 1x NVIDIA RTX 4090 (24gb DDR6X VRAM)

The latter option is significantly faster, capable of processing stacks up to 25gb large in under an hour and a half with default parameters given the extensive parallelization capacity of NVIDIA's CUDA framework.

#### Neuron reconstruction

***Volumetric data generation***

Matlab functions and scripts to generate volumetric labels from manual traces using a topology preserving fast marching algorithm.
Original github repo [here](https://github.com/rhngla/topo-preserve-fastmarching).

***Segmentation***

Code for training a neuron network model to perform a multi-class segmentation of neuronal arbors as well as for running inference using trained models is in the `pytorch_segment` section of this repository.
Original github repo [here](https://github.com/jgornet/NeuroTorch).

***Postprocessing***

Code for postprocessing including relabeling to improve axon/dendrite node assignment of the initial reconstruction and merging segments is in the `postprocessing` section of this repository.

***Neuron reconstruction pipeline***

Automated pipeline combines pre-processing raw images, segmentation of raw image stack into soma/axon/dendrite channels, post-processing, and conversion of images to swc file. This code, and [a small example](https://github.com/ogliko/patchseq-autorecon/blob/master/pipeline/example_pipeline.sh) can be found under the `pipeline` section of this repository. The example's maximum intensity projection (mip) is seen [here](https://github.com/ogliko/patchseq-autorecon/blob/master/pipeline/Example_Specimen_2112/example_specimen.PNG) 
 
***SWC Post-Processing***

Code for creating swc post-processing workflows can be found here:
 https://github.com/MatthewMallory/morphology_processing 

#### Data analysis

 - generating axonal and dendritic arbor density representations (`analysis/arbor_density`)
 - cell type classification using arbor density representations and morphometric features (`analysis/cell_type_classification`)
 - sparse feature selection (`analysis/sparse_feature_selection`)