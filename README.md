## Radynversion: Learning to Invert a Solar Flare Atmosphere with Invertible Neural Networks

This repository contains the code for Radynversion, an Invertible Neural Network (INN) based tool that infers solar atmospheric properties during a solar flare, based on a model trained on RADYN simulations (references not limited to Carlsson & Stein 1992, 1995, 1999, and many more..., Allred et al. 2005, 2015). RADYN is a 1D non-equilibrium radiation hydrodynamic model with good optically thick radiation treatment (under the assumption of complete redistribution). It does not consider magnetic effects.

Our INN is trained to learn an approximate bijective mapping between the atmospheric properties of electron density, temperature, and bulk velocity (all as a function of altitude), and the observed H\alpha and Ca II 8542 line profiles. As information is lost in the forward process of radiation transfer, this information is injected back into the model during the inverse process by means of a latent space. Thanks to the training, this latent space can now be filled using an n-dimensional unit Gaussian distribution, where n is the dimensionality of the latent space.

The bijectivity of the model is assured by the building blocks of the INN, the affine coupling layers. By splitting the data into two streams, these blocks combine four arbitrarily complex non-invertible functions and apply these to the input data in a reversible manner. 

This processs, and its validation are described in the paper: Osborne, Armstrong, and Fletcher (2019) `arXiv link here when public`. The code associated with this paper lives in this repository, but the Radynversion tool will in time be merged into the 
[RadynPy](https://github.com/Goobley/radynpy) python module.

For an example of the model in action the `single_pixel_inversion_example.ipynb` notebook is the recommended place to start. It uses a library of functions defined in `utils.py` You will also need the model weights, which are available on the Github releases page for this project.
To look at the training of the model, the reader is directed to `Radynversion.ipynb`, which calls functions from `Inn2.py` and `Loss.py`. To train your own variant of the model you can use our data extracted from the F-CHROMA RADYN simulations grid, also available on the releases page (the _ridiculously_ named `DoublePicoGigaPickle50.pickle`) or you can generate your own from Radyn simulations via `ExportSimpleLineBlobForTraining.py`. At the very least, the paths in this last script will need modifying for your system and simulation set.

The two main notebooks specify their required packages. The combined requirements are:
- `Python 3`
- `numpy`
- `scipy`
- `matplotlib`
- `pytorch` (currently `0.4.1`, but should also be compatible with `1.0`, though I have yet to check).
- `astropy`
- `scikit-image`
- `palettable` (optional, only required for colourmaps, but no fail-safes in the code if not preset).
- `crisPy` ([available here](https://github.com/rhero12/crisPy)
- `FrEIA` ([available here](https://github.com/VLL-HD/FrEIA)
- `RadynPy` ([available here](https://github.com/Goobley/radynpy), needed for loading RADYN outputs, so essential for making your own training set, not currently required otherwise, though Radynversion will eventually be accessible as a RadynPy module).
Some of these packages will also have their own requirements, but your package manager should hopefully be able to figure most of that out!

Developed by Chris Osborne & John Armstrong, University of Glasgow, Astronomy and Astrophysics (2018-2019). MIT License.
Please drop us an email with comments, suggestions etc. Contact address `c.osborne.1 [at] research [dot] gla [dot] ac [dot] uk`.
