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
│   ├── simulation.ipynb          # Main simulation workflow
│   ├── visualization.ipynb       # Plaque & infection visualizations
│   └── parameter_analysis.ipynb  # Sensitivity / parameter studies
│
├── utils.py                # Utility functions for random seeds, diffusion, statistics
├── model.py                # Entry point to run simulations
│
├── data/
│   ├── params.json         # Parameter sets used in experiments
│   ├── snapshots/          # Saved simulation states (optional)
│   └── measurements.csv    # Experimental reference data
│
├── result/
│   ├── plaque_growth.png
│   ├── infection_dynamics.png
│   └── heatmap_entry_efficiency.png
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

## ▶️ Running a Simulation

```bash
python model.py --steps 1000 --npc1_variant WT
```

### Optional Flags

| Flag | Description |
|------|-------------|
| `--steps` | Number of simulation steps |
| `--grid_size` | Size of the cell monolayer grid |
| `--npc1_variant` | NPC1 genotype (e.g., WT, SNP1, SNP2) |
| `--save_snapshots` | Save intermediate images |
| `--seed` | Random seed |

Example:

```bash
python model.py --steps 1500 --grid_size 250 --npc1_variant SNP1 --save_snapshots
```

---

## 📓 Using Jupyter Notebooks

```bash
jupyter notebook notebooks/simulation.ipynb
```

Available notebooks:

- **simulation.ipynb** – full plaque growth simulation  
- **visualization.ipynb** – heatmaps, plaque radius tracking, infection patterns  
- **parameter_analysis.ipynb** – parameter sweeps and sensitivity analysis  

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

## 📊 Example Outputs

### **Plaque Growth Over Time**
![plaque_growth](result/plaque_growth.png)

### **Infection Dynamics**
![infection_dynamics](result/infection_dynamics.png)

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
