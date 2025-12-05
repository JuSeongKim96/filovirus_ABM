# 🧬 filovirus_ABM  
Agent-based model (ABM) implementation for simulating **filovirus entry efficiency** and **plaque growth dynamics**, used in the study:

> **Evaluating the impact of NPC1 single nucleotide polymorphisms on entry efficiency of filoviruses in vitro: Agent-based model approach**  
> *Juseong Kim, Kwang Su Kim, Ayato Takada, Yusuke Asai, Shingo Iwami, Seung-Woo Son, Mi Jin Lee*  
> **Journal of Theoretical Biology (2025)**  
> DOI: https://doi.org/10.1016/j.jtbi.2025.112315

---

## 📘 Overview

This repository reproduces an **agent-based simulation** of filovirus infection at the single-cell level.  
The model incorporates:

- NPC1 genotype–dependent viral entry efficiency  
- Virus diffusion and spatial plaque expansion  
- Cell state transitions (susceptible → infected → dead)  
- Replication of in vitro plaque assay behavior

The implementation supports both **batch simulations** and **interactive Jupyter notebooks** for visualization and analysis.

---

## 📁 Repository Structure

```
filovirus_ABM/
│
├── notebooks/              # Core ABM implementations
│   ├── Sim.ipynb          # Main simulation workflow
│   ├── model.py       # infection process
│   └── utils.py  # Utility functions
│
├── data/
│   ├── Angola         # Marburg virus radius data
│   ├── Zaire          # Ebola virus radius data
│   ├── Triangle       # Optimal beta in triangle lattice
│   └── Rectangle      # Optimal beta in rectangle lattice
│
├── result/
│   ├── Fig3.png                  # Entry efficiency
│   ├── best_beta_fit             # Best beta fitting radius 
│   ├── Zaire_error_function.png  # Optimal beta
│   └── Angola_error_function.png # Optimal beta
│
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/JuSeongKim96/filovirus_ABM.git
cd filovirus_ABM

pip install -r requirements.txt
```

*Python ≥ 3.8 recommended.*

---

## 📓 Using Jupyter Notebooks

```bash
jupyter notebook notebooks/simulation.ipynb
```

---

## 🧠 Model Description

### **Cell states**
- Susceptible  
- Infected  
- Dead  

### **Processes**
- **Viral entry** probability modulated by NPC1 genotype  
- **Virus diffusion** across neighboring cells  
- **Local plaque formation**  
- **Tracking infection fronts** over discrete time steps  

### **Simulation workflow**
1. Initialize grid of host cells  
2. Seed infection at t = 0  
3. At each time-step:  
   - Compute genotype-specific entry probability  
   - Infect neighbors & propagate virus  
   - Update cell states  
   - Measure plaque radius  
4. Save results or snapshots  

---

## 📚 Citation

If you use this repository, please cite:

```
Kim J., Kim K.S., Takada A., Asai Y., Iwami S., Son S.-W., Lee M.J.
Evaluating the impact of NPC1 single nucleotide polymorphisms on entry efficiency of filoviruses in vitro: Agent-based model approach.
Journal of Theoretical Biology, 2025.
https://doi.org/10.1016/j.jtbi.2025.112315
```

---

## 👥 Contributors

- **Juseong Kim**  
- **Kwang Su Kim**  
- **Ayato Takada**  
- **Yusuke Asai**  
- **Shingo Iwami**  
- **Seung-Woo Son**  
- **Mi Jin Lee**

---

## 📬 Contact

```
Email: ju.seong.kim.1996@gmail.com  
GitHub Issues: https://github.com/JuSeongKim96/filovirus_ABM/issues
```

---

## 🔐 License

MIT License
