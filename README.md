# Master_Thesis
Master Thesis on "Small Molecule Generation and Optimization: A GNN-VAE Approach submitted" by Nesara Belakere Lingarajaiah Matr. Nr.: 7002967 Saarbrücken January 2024

## Abstract
Deep learning has demonstrated remarkable efficacy across diverse domains encompass-
ing acoustics, imagery, and natural language processing. Drawing inspiration from these
achievements, researchers are presently harnessing generative model methodologies in
pursuit of de novo drug design - an endeavor long held as the pinnacle aspiration within
the realm of pharmaceutical discovery. The process of drug design is fraught with signif-
icant resource and time expenses. Generative deep learning techniques are progressively
leveraging extensive repositories of biochemical data and augmented computational
capabilities to forge new methodologies conducive to the enhancement of drug discovery
and optimization.
While early approaches were reliant on Simplified molecular-input line-entry system
(SMILES) strings, contemporary methodologies have gravitated towards employing
molecular graphs as a more innate means to represent chemical entities. Yet, the integra-
tion of deep learning into the domain of graph data presents intricate challenges, owing
to the unique nature of graph structures and in graph based methods we wish to be
invariant to reordering of nodes which is a challenging task. Recent research endeavors
have been substantially committed to incorporating deep learning techniques into the
sphere of graph data, yielding commendable advancements in the domains of graph
analysis and generation methodologies.
To address this challenge, this thesis presents an innovative approach that integrates a
Graph Neural Networks (GNN)-based objective function i.e the Similarity Loss for graph
comparison, enabling the generation of molecular structures based on graphs. More
specifically, this method employs a variational autoencoder framework, showcasing its
capacity to effectively generate molecular structures. Through the synergy of generative
models and the incorporation of a GNN-based objective function, our approach success-
fully alleviates the constraints associated with graph representations in the context of de
novo drug design.
Furthermore, this thesis extends its analysis and evaluation of the GNN-based Similarity
Loss by delving into advanced alternative methods. It also investigates techniques for
optimizing and exploring the latent space to create molecules with substantial property
alterations while making minimal modifications. By manipulating the latent space, we
unlock the potential to uncover molecules with specific attributes or to explore uncharted
areas within the chemical space that may possess distinctive properties.

## Architecture
![general_vae_architecture](https://github.com/nesara/Master_Thesis/assets/29191510/9a2cad7e-da46-4826-a860-0e6c2113e787)

## Molecule Reconstructions on Zinc Dataset
![output1](https://github.com/nesara/Master_Thesis/assets/29191510/fe13a959-65c4-4ca9-9bd7-04185d659489)
![output10](https://github.com/nesara/Master_Thesis/assets/29191510/39cf49ff-098f-4245-ba05-7cfc41db21ec)

## Nearby Molecules in the Latent Space
![input2](https://github.com/nesara/Master_Thesis/assets/29191510/41dbf7c2-7c90-4d8f-ae9c-037242d0ed58)
![output2](https://github.com/nesara/Master_Thesis/assets/29191510/06332b1f-e333-4385-ba14-7c40da21fc91)


