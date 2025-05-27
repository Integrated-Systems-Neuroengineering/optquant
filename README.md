# Where to cut?

Many proposed in-memory-computing systems use memristive crossbars to compute matrix-vector products over discrete domains, e.g. binary or ternary inputs and weights. This yields analog output values, e.g. voltages, that are distributed around discrete values that can cover a wide range. But as implied by the central limit theorem, typical results are highly concentrated only in a small central region. 
Lossless quantization of the entire range requires costly high-precision analog-to-digital converters (ADCs), but this is not necessary for many applications like neural network accelerators. Instead, an ADC with lower resolution that is linear only in this central region and saturates outside it can achieve almost full accuracy at only a fraction of the cost. 

In this work, presented at [ISCAS 2025](https://2025.ieee-iscas.org/full-program), we derive optimal parameters for such an ADC. 

# The repository

```markdown
├── 📁 adim                                 → Code to compute parameters for **a**daptive **d**iscrete **i**n-**m**emory systems
├── 📁 experiments                          → Scripts (JuPyTer notebooks) to generate the corresponding figures shown in the paper
│   ├── 📜 overview.ipynb                   → Shows an overview of the system
│   ├── 📜 noise.ipynb                      → Showns the dependence of the optimal parameterization on noise
│   ├── 📜 optimum.ipynb                    → Illustrates the optimum parameterization for various ADC resolutions
│   ├── 📜 range.ipynb                      → Analyzes the optimal range of ADCs for various conditions
├── 📁 figures                              → The figures shown in the paper/poster
│   ├── 🖻 overview.{pdf,png,svg}           → Shows an overview of the system
│   ├── 🖻 noise.{pdf,png,svg}              → Showns the dependence of the optimal parameterization on noise
│   ├── 🖻 optimum.{pdf,png,svg}            → Illustrates the optimum parameterization for various ADC resolutions
│   ├── 🖻 range.{pdf,png,svg}              → Analyzes the optimal range of ADCs for various conditions
├── 📄 requirements.txt                     → required python packages (install via `pip`)
├── 📄 poster.pdf                           → the poster
├── 📄 paper.pdf                            → the paper
├── 📄 README.md                            → this file
└── 📄 LICENSE                              → license information
```
