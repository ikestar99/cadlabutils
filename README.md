## Cadwell Lab Patch-seq Morphology Reconstruction Pipeline
Ike Ogbonna (computational, Ike.Ogbonna@ucsf.edu)
Naomi Crane (computational/experimental, ncrane6@jh.edu)
Kevin Lee, PhD (experimental, Kevin.Lee2@ucsf.edu)
Cathryn Cadwell, MD, PhD (principal investigator, Cathryn.Cadwell@ucsf.edu)

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


<!--
Hey, thanks for using the awesome-readme-template template.  
If you have any enhancements, then fork this project and create a pull request 
or just open an issue with the label "enhancement".

Don't forget to give this project a star for additional support ;)
Maybe you can mention me or this repo in the acknowledgements too
-->

<!--
This README is a slimmed down version of the original one.
Removed sections:
- Screenshots
- Running Test
- Deployment
- FAQ
- Acknowledgements
-->

<div align="center">

  <img src="assets/logo.png" alt="logo" width="200" height="auto" />
  <h1>Awesome Readme Template</h1>
  
  <p>
    An awesome README template for your projects! 
  </p>

  
<!-- Badges -->
<p>
  <a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Louis3797/awesome-readme-template" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/Louis3797/awesome-readme-template" alt="last update" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/network/members">
    <img src="https://img.shields.io/github/forks/Louis3797/awesome-readme-template" alt="forks" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/stargazers">
    <img src="https://img.shields.io/github/stars/Louis3797/awesome-readme-template" alt="stars" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/issues/">
    <img src="https://img.shields.io/github/issues/Louis3797/awesome-readme-template" alt="open issues" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Louis3797/awesome-readme-template.svg" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/Louis3797/awesome-readme-template/">View Demo</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template">Documentation</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template/issues/">Request Feature</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents

- [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [Features](#features)
  * [Color Reference](#color-reference)
  * [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Run Locally](#run-locally)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
  * [Code of Conduct](#code-of-conduct)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

<!-- About the Project -->
## About the Project

<div align="center"> 
  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />
</div>


<!-- TechStack -->
### Tech Stack

<details>
  <summary>Client</summary>
  <ul>
    <li><a href="https://www.typescriptlang.org/">Typescript</a></li>
    <li><a href="https://nextjs.org/">Next.js</a></li>
    <li><a href="https://reactjs.org/">React.js</a></li>
    <li><a href="https://tailwindcss.com/">TailwindCSS</a></li>
  </ul>
</details>

<details>
  <summary>Server</summary>
  <ul>
    <li><a href="https://www.typescriptlang.org/">Typescript</a></li>
    <li><a href="https://expressjs.com/">Express.js</a></li>
    <li><a href="https://go.dev/">Golang</a></li>
    <li><a href="https://nestjs.com/">Nest.js</a></li>
    <li><a href="https://socket.io/">SocketIO</a></li>
    <li><a href="https://www.prisma.io/">Prisma</a></li>    
    <li><a href="https://www.apollographql.com/">Apollo</a></li>
    <li><a href="https://graphql.org/">GraphQL</a></li>
  </ul>
</details>

<details>
<summary>Database</summary>
  <ul>
    <li><a href="https://www.mysql.com/">MySQL</a></li>
    <li><a href="https://www.postgresql.org/">PostgreSQL</a></li>
    <li><a href="https://redis.io/">Redis</a></li>
    <li><a href="https://neo4j.com/">Neo4j</a></li>
    <li><a href="https://www.mongodb.com/">MongoDB</a></li>
  </ul>
</details>

<details>
<summary>DevOps</summary>
  <ul>
    <li><a href="https://www.docker.com/">Docker</a></li>
    <li><a href="https://www.jenkins.io/">Jenkins</a></li>
    <li><a href="https://circleci.com/">CircleCLI</a></li>
  </ul>
</details>

<!-- Features -->
### Features

- Feature 1
- Feature 2
- Feature 3

<!-- Color Reference -->
### Color Reference

| Color             | Hex                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Primary Color | ![#222831](https://via.placeholder.com/10/222831?text=+) #222831 |
| Secondary Color | ![#393E46](https://via.placeholder.com/10/393E46?text=+) #393E46 |
| Accent Color | ![#00ADB5](https://via.placeholder.com/10/00ADB5?text=+) #00ADB5 |
| Text Color | ![#EEEEEE](https://via.placeholder.com/10/EEEEEE?text=+) #EEEEEE |


<!-- Env Variables -->
### Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`

<!-- Getting Started -->
## Getting Started

<!-- Prerequisites -->
### Prerequisites

This project uses Yarn as package manager

```bash
 npm install --global yarn
```

<!-- Installation -->
### Installation

Install my-project with npm

```bash
  yarn install my-project
  cd my-project
```


<!-- Run Locally -->
### Run Locally

Clone the project

```bash
  git clone https://github.com/Louis3797/awesome-readme-template.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  yarn install
```

Start the server

```bash
  yarn start
```


<!-- Usage -->
## Usage

Use this space to tell a little more about your project and how it can be used. Show additional screenshots, code samples, demos or link to other resources.


```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```

<!-- Roadmap -->
## Roadmap

* [x] Todo 1
* [ ] Todo 2

<!-- Contributing -->
## Contributing

<a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" />
</a>


Contributions are always welcome!

See `contributing.md` for ways to get started.


<!-- Code of Conduct -->
### Code of Conduct

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)


<!-- License -->
## License

Distributed under the no License. See LICENSE.txt for more information.


<!-- Contact -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/Louis3797/awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)

<!-- Acknowledgments -->
## Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [Shields.io](https://shields.io/)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [Emoji Cheat Sheet](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md#travel--places)
 - [Readme Template](https://github.com/othneildrew/Best-README-Template)