# gan-pytorch

Learning and implementation of GAN networks in PyTorch.

## Important Links 

- [Issues while training GANs](https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b)
* Non-convergence: the model parameters oscillate, destabilize and never converge,
* Mode collapse: the generator collapses which produces limited varieties of samples,
* Diminished gradient: the discriminator gets too successful that the generator gradient vanishes and learns nothing,
* Unbalance between the generator and discriminator causing overfitting, and
* Highly sensitive to the hyperparameter selections.   

- [Hacks to train GANs](https://github.com/soumith/ganhacks)

## Networks 

[1] [DCGAN: Deep Convolutional GAN](/dcgan)
