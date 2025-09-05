<div align="center">
  <p align="center">
    <img src="assets/logo-day.png#gh-light-mode-only" alt="logo-day" width="600" height="auto" />
    <img src="assets/logo-night.png#gh-dark-mode-only" alt="logo-night" width="600" height="auto" />
  </p>
  
  <p>
    Python utilities for data analysis pipelines across the Cadwell Lab at UCSF 
  </p>

  
<!-- Badges -->
<p>
  <a href="https://github.com/ikestar99/cadlabutils/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/ikestar99/cadlabutils" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/ikestar99/cadlabutils" alt="last update" />
  </a>
  <a href="https://github.com/ikestar99/cadlabutils/issues/">
    <img src="https://img.shields.io/github/issues/ikestar99/cadlabutils" alt="open issues" />
  </a>
  <a href="https://github.com/ikestar99/cadlabutils/LICENSE">
    <img src="https://img.shields.io/github/license/ikestar99/cadlabutils.svg" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/ikestar99/cadlabutils">Documentation</a>
  <span> · </span>
    <a href="https://github.com/ikestar99/cadlabutils/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/ikestar99/cadlabutils/issues/">Request Feature</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# Table of Contents


<details>
<summary>Installation</summary>

- [Environment](#environment)  
- [Install From Source](#install-from-source)  
- [Dependencies](#dependencies)  

</details>

- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
  

[//]: # (<div align="center"> )
[//]: # (  <img src="https://placehold.co/600x400?text=Your+Screenshot+here" alt="screenshot" />)
[//]: # (</div>)


<!-- Installation -->
## Installation


<!-- env prep -->
### Environment
Create a dedicated environment for your project.

```bash
# conda
conda create -n myenv python=3.13.5
conda activate myenv

# pip/venv:
python3.13 -m venv myenv
source myenv/bin/activate   # Linux/macOS
myenv\Scripts\activate      # Windows PowerShell
```

Alternatively, activate an existing environment for an established project.
```bash
# conda
conda activate myenv

# pip/venv:
source myenv/bin/activate   # Linux/macOS
myenv\Scripts\activate      # Windows PowerShell
```


<!-- Install From Source -->
### Install From Source

Install `cadlabutils` directly from GitHub, w/o optional dependencies.
```bash
# Minimal installation
pip install "git+https://github.com/ikestar99/cadlabutils.git"

# Array manipulation tools
pip install "git+https://github.com/ikestar99/cadlabutils.git#egg=cadlabutils[arrays]"

# File manipulation tools
pip install "git+https://github.com/ikestar99/cadlabutils.git#egg=cadlabutils[files]"

# Pytorch utilities
pip install "git+https://github.com/ikestar99/cadlabutils.git#egg=cadlabutils[learning]"

# Complete install
pip install "git+https://github.com/ikestar99/cadlabutils.git#egg=cadlabutils[dev]"
```

Clone repo and install in editable configuration for local development
```bash
# clone project in current working directory
git clone https://github.com/ikestar99/cadlabutils.git
cd ./cadlabutils
pip install -e ".[dev]"
```


<!-- Dependencies -->
### Dependencies

cadlabutils includes a set of optional dependencies for:
- `[arrays]`: array operations
- `[files]`: specific file extensions
- `[learning]`: deep learning with [`pytorch`](https://pytorch.org/)
- See [`pyproject.toml`](https://github.com/ikestar99/cadlabutils/blob/main/pyproject.toml) for specific packages
in each set of optional dependencies.

Some included utilities and dependencies aren't applicable to all projects, such as
[`zarr`](https://zarr.readthedocs.io/en/stable/). These imports are limited to the submodules that use them directly.
Thus, a project that requires [`h5py`](https://www.h5py.org/) but not [`zarr`](https://zarr.readthedocs.io/en/stable/)
need not install all packages included in the `[files]` optional dependency build.

Instead, use a minimal install of `cadlabutils` and import only the modules you need:
```python
import cadlabutils.files.h5s as cdu_h5


with cdu_h5.File("/tmp/path.h5", mode="w") as data:
    dset = cdu_h5.make_dataset(data, name="dataset_0", shape=(100, 100), dtype=int)
```

Other modules contain dependencies that are often used in combination, and thus share a single namespace and should be
installed together.
```python
from pathlib import Path

import cadlabutils.learning as cdu_l


model = cdu_l.SNEMI3D(c_i=1, c_o=3)
checkpoint_pth = Path("/Path/to/saved/weights.safetensors")
model, _ = cdu_l.load(checkpoint_pth, model, device=cdu_l.get_device(None))
```


<!-- Usage -->
## Usage

TBD


<!-- Contributing -->
## Contributing

<a href="https://github.com/ikestar99/cadlabutils/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ikestar99/cadlabutils" />
</a>

TBD


<!-- License -->
## License

Distributed under the Apache License. See LICENSE.txt for more information.


<!-- Contact -->
## Contact

Ike Ogbonna, MS (Ike.Ogbonna@ucsf.edu - ikestar99@hotmail.com)
Cathryn Cadwell, MD, PhD (Cathryn.Cadwell@ucsf.edu)

Project Link: [https://github.com/Louis3797/awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)


<!-- Acknowledgments -->
## Acknowledgements

logo art:
 - Edward Valenzuela, MS (edward.valenzuela@ucsf.edu)

Templates/badges:
 - [Shields.io](https://shields.io/)
 - [awesome-readme-template](https://github.com/Louis3797/awesome-readme-template)