# NeuronDynamicsML
Modeling neuronal voltage dynamics using Physics-Informed Neural Networks and Sparse Identification of Nonlinear Dynamics.
# NeuronDynamics-PINN-SINDy

This repository contains the Colab notebooks and documentation for the project:
**"Data-Driven and Physics-Informed Machine Learning for Chaotic Neuronal Dynamics"**.

We apply **Physics-Informed Neural Networks (PINNs)** and **Sparse Identification of Nonlinear Dynamics (SINDy)** to learn interpretable biophysical models from intracellular voltage recordings of zebra finch neurons.

---

## 📁 Project Structure

- `colabs/`: All Google Colab notebooks used in the project
- `data/`: Intracellular voltage and injected current traces
- `figures/`: Key plots exported from the notebooks
- `README.md`: This file
- `requirements.txt`: Python libraries, if running locally

---

## 🧰 Setup Instructions

You can run the notebooks directly in [Google Colab](https://colab.research.google.com/).  
Click any of the links below:

- 📓 [PINN - Single Ion Channel](colabs/PINN_SingleCurrent.ipynb)
- 📓 [PINN - SINDy_and_Full Ionic Model](colabs/PINN_ExtendedModel.ipynb)

> ⚠️ If opening from GitHub, click the **"Open in Colab"** badge or copy the notebook URL into Colab manually.

---

## 📦 Optional Local Setup

If you'd like to run the notebooks locally, create a virtual environment and install:

```bash
pip install -r requirements.txt
