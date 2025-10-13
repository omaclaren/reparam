# Mathematical models with explicit identifiable parameter combinations

**Research has identified 40+ mathematical models (5-30 parameters) with explicitly reported identifiable parameter combinations from structural identifiability analysis.** These span systems biology, pharmacology, epidemiology, and engineering, with clear mathematical expressions for combinations like k₁₂·k₂₁, β/N, and Rtot·ke. The most comprehensive benchmarking study analyzed **25 model variants** with detailed identifiability results, while major software tools (DAISY, STRIKE-GOLDD, GenSSI, SIAN) provide worked examples with explicit combination expressions. Models range from simple 2-compartment pharmacokinetic systems to complex 34-state metabolic networks, with identifiable combinations typically taking forms of products, ratios, and sums of unidentifiable parameters.

## Benchmark models from major comparison studies

The definitive benchmarking paper by Rey Barreiro & Villaverde (2023) in *Bioinformatics* compared 13 identifiability analysis tools on **25 variants of 21 models** specifically selected to test computational methods. This collection represents the current standard for testing identifiability algorithms.

**Two-compartment Michaelis-Menten model** is the canonical small example. With **5 parameters** (k₀₁, k₀₂, k₁₂, k₂₁, V₁), the model is non-identifiable when measuring only compartment 1. Meshkat et al. (2014, *PLOS ONE* 9(10):e110261) showed that V₁ is uniquely identifiable while the identifiable combinations are **k₁₂·k₂₁** (product), **k₀₁ + k₂₁** (sum), and **k₀₂ + k₁₂** (sum). These combinations can be used to reparameterize the model by defining k₁₁ = -(k₀₁ + k₂₁) and k₂₂ = -(k₁₂ + k₀₂), replacing four unidentifiable parameters with two identifiable sums and one identifiable product. The COMBOS web application (http://biocyb1.cs.ucla.edu/combos) provides interactive analysis of this model.

**Three-compartment mammillary model** extends this to **7 parameters** (k₀₂, k₀₃, k₁₂, k₁₃, k₂₁, k₃₁, V₁) with sampling from the central compartment. Analysis shows V₁ is uniquely identifiable, while **k₁₁ = -(k₀₂ + k₀₃ + k₁₂ + k₁₃)** is uniquely identifiable. The combinations **k₁₂·k₂₁** and **k₁₃·k₃₁** are each locally identifiable with 2 solutions. Similarly, k₂₂ = -(k₂₁ + k₀₂) and k₃₃ = -(k₃₁ + k₀₃) have 2 solutions each. This demonstrates the typical pattern in compartmental models where volume parameters are identifiable but rate constants form product and sum combinations.

**HIV dynamics models** appear extensively in identifiability literature as test cases. The classic Perelson model has **8 parameters** (s, d, β, a, p, c, δ, ε) representing target cells, infected cells, and virions. When measuring only viral load, Meshkat et al. (2014) showed that d (death rate) is uniquely identifiable, while **β·p** (infection rate times viral production) and **s·δ·a** are uniquely identifiable combinations. The parameters c (clearance) and ε (drug efficacy) are locally identifiable with 2 solutions each. This model appears in COMBOS, DAISY examples, and SIAN benchmarks, making it one of the most thoroughly analyzed identifiability test cases.

## Pharmacodynamic receptor models reveal systematic patterns

Janzén et al. (2016, *Frontiers in Physiology* 7:590) performed systematic identifiability analysis on **16 pharmacodynamic models** ranging from 5-8 parameters each. Every single model exhibited the same identifiability pattern: the total receptor amount (Rtot) and related parameters form consistent combinations.

The fundamental identifiable combinations across all 16 models are **Rtot × ke** (product of total receptors and signal transduction rate) and **Rtot / RC₅₀** (called the transducer ratio τ). These combinations replace three unidentifiable parameters (Rtot, ke, RC₅₀) with two identifiable ones. This work demonstrates that certain parameter combinations recur systematically across mechanistically similar models.

**Model 10** (effect compartment with operational model and linear transduction) has **6 parameters**: ke0, kon, koff, Rtot, RC50, ke. Analysis confirmed that fixing Rtot to a reference value (typically 1 or 100%) renders all other parameters uniquely identifiable. The traditional pharmacology parameters Emax and EC50 can be derived from the identifiable combinations: Emax = ke × Rtot and EC50 = Kd × RC50. Code is available through the IMPACT project demonstrating application to AZD1305 drug effects on QT interval prolongation.

Zhu et al. (2018, *British Journal of Pharmacology*) extended this to biased agonism at GPCRs, finding that only the transduction coefficient **R = τ/KA** is practically identifiable despite structural identifiability of individual efficacy parameters. Their recommendation to fix either KA (affinity) or Em (maximal response) aligns with the general principle that fixing one parameter in a correlated pair enables identification of the others.

## Systems biology pathway models from 13-34 states

Villaverde et al. (2016, *PLOS Computational Biology* 12(10):e1005153) introduced the STRIKE-GOLDD software with seven detailed case studies, several in the target parameter range. These examples demonstrate how structural identifiability analysis scales to realistic biological models.

**JAK/STAT signaling pathway** models IL13-induced signaling in lymphoma with **13 states and 23 parameters** (θ₁ through θ₂₃). The analysis identified 5 unidentifiable parameters: θ₁₁, θ₁₅, θ₁₇, θ₂₁, θ₂₂. Critically, the observability matrix kernel revealed the identifiable combination **Φ = θ₁₇ × θ₂₂**. By fixing θ₁₁ a priori (which can be done using prior biological knowledge), the model with 21 unknown parameters becomes structurally identifiable. The mathematical expressions for combinations were derived systematically from the null space of the extended observability matrix. Code is available at https://github.com/afvillaverde/strike-goldd with full model specifications.

**NF-κB regulatory module** from Lipniacki et al. (2004) has **15 states and 29 parameters** modeling immune response signaling. Analysis found 25 parameters identifiable with 5 unidentifiable: c₁c, c₂c, c₃c, c₄, k₂. The first three (c₁c, c₂c, c₃c) appear only in the equation for state x₁₅, which is unobserved and disconnected from observed states—they cannot possibly affect model output. More interestingly, c₄ and k₂ are related such that fixing either renders the other identifiable. The solution is to fix c₁c, c₂c, c₃c and either c₄ or k₂, leaving 25 identifiable parameters. A reduced 13-parameter version (Jaruszewicz-Błońska et al., 2023, *PLOS One*) achieved both structural and practical identifiability with 5 observables, demonstrating how model simplification improves identifiability.

**Arabidopsis thaliana circadian clock** (Locke et al., 2005) models genetic control of circadian rhythm with **7 states and 27 parameters** including Hill coefficients. Measuring only 2 mRNA concentrations (x₁, x₄) while protein concentrations remain unmeasured creates identifiability problems. Villaverde's decomposition approach identified 11 individually identifiable parameters: a, k₁, k₄, m₁, m₄, n₁, n₂, q₂, r₁, r₂, r₄. The solution is to fix 5 degradation constants (k₂, k₃, k₅, k₆, k₇) based on literature values, rendering the remaining 23 parameters identifiable. Alternatively, measuring all 7 states makes all 27 parameters identifiable, showing how experimental design choices directly impact identifiability.

**MAPK cascade** with mixed feedback (Nguyen et al., 2015) models three-layer signaling with **14 parameters**: k₁, k₂, k₃, k₄, k₅, k₆, s₁t, s₂t, s₃t, K₁, K₂, n₁, n₂, α. The model requires all 3 phosphorylation states measured for full identifiability. If x₁ is not measured, then k₃ and s₁t are unidentifiable. If x₂ is not measured, k₅ and s₂t are unidentifiable. If x₃ is not measured, K₁, K₂, and s₃t are unidentifiable. This demonstrates how observability of specific states determines which parameter subsets can be identified.

## Large-scale metabolic model demonstrates scalability

**Chinese hamster ovary (CHO) cell metabolism** for fed-batch protein production represents the upper size limit successfully analyzed. The model has **34 metabolite states, 13 measured outputs, and 117 parameters** covering glycolysis, TCA cycle, amino acid metabolism, and electron transport using lin-log kinetics. Villaverde et al. (2015, BioPreDyn-bench B4) found 4 initially unidentifiable elasticity parameters: e₅₄, e₅₅, e₆₂, e₆₄. However, the combinations **e₅₄ + e₅₅** and **e₆₂ + e₆₄** are identifiable, reducing the parameter count to 115. Further analysis determined 97 parameters are identifiable. Fixing 6 specific parameters (p₂₈, p₇₂, p₇₇, p₁₀₁, p₁₀₅, p₁₁₅) based on prior knowledge renders all remaining parameters identifiable. This demonstrates that identifiability analysis is computationally feasible even for genome-scale models, though the identifiable combinations become more complex.

## Enzyme kinetics models highlight practical challenges

McGuinness et al. (2024, *BMC Bioinformatics*) analyzed CD39/NTPDase1 with substrate competition for ATP→ADP→AMP reactions. Despite having only **4 parameters** (Vmax1, Vmax2, Km1, Km2), substrate competition creates severe parameter correlations. The Michaelis constants appear as reciprocal ratios in both rate equations: Km1/Km2 and Km2/Km1. This causes correlation of 0.99 between Km parameters, making simultaneous estimation essentially impossible despite structural identifiability.

Their solution—**reaction isolation strategy**—estimates ADPase parameters (Vmax2, Km2) separately using ADP-only data, then estimates ATPase parameters (Vmax1, Km1) using ATP data with ADPase parameters fixed. This sequential approach achieved 100% parameter recovery versus 0.05-12% with naive simultaneous estimation. Code is available at https://github.com/AndrewDMarquis/CD39-Enzyme-Kinetics. This example demonstrates that structural identifiability does not guarantee practical identifiability—experimental design and estimation strategy matter critically.

## Epidemiological models with explicit mathematical expressions

Chowell et al. (2023, *Journal of Mathematical Biology* 87(6):79) provided tutorial-level detail on identifiability of epidemic models with code at https://github.com/sushmadahal/Identifiability. Their systematic analysis covers models of increasing complexity, all explicitly stating identifiable parameter combinations.

**Basic SEIR model** with **4 parameters** (N, β, k, γ) observing only cumulative incidence yields three identifiable combinations: k, γ, and **β/N**. The ratio β/N (per-capita transmission rate) is identifiable but not the individual parameters. With known initial conditions, all four parameters become uniquely identifiable. This demonstrates the role of initial condition knowledge in breaking parameter correlations.

**SEIR with symptomatic/asymptomatic infections** adds parameter ρ (fraction symptomatic) for **5 total parameters** (N, β, k, γ, ρ). Without initial conditions, only k, γ, and β/N are identifiable—the additional parameter ρ remains unidentifiable from case data alone. With known initial conditions, all 5 parameters are uniquely identifiable. This shows that model complexity can outpace available data, requiring additional measurements or prior knowledge.

**SEIR with distinct asymptomatic transmission** uses **6 parameters** (N, βA, βI, k, ρ, γ) allowing different transmission rates for asymptomatic (βA) and symptomatic (βI) infections. The identifiable combinations become **Nρ = N̂ρ̂** and **βIρ + βA(1-ρ) = β̂Iρ̂ + β̂A(1-ρ̂)**, showing that the population-weighted transmission rate is identifiable but not individual components. With known initial conditions, all 6 parameters are identifiable.

**SEIR with disease-induced deaths** has **4 parameters** (β, k, γ, δ) where δ is disease-specific mortality. Observing only new cases yields three combinations: k, **β/δ**, and **γ+δ**. The ratio of transmission to mortality (β/δ) and the sum of recovery plus death rates (γ+δ) are identifiable. Observing both new cases AND deaths makes all 4 parameters individually identifiable, demonstrating how multiple data streams resolve identifiability.

**Vector-borne disease model** with **6 parameters** (Λυ, μυ, βυ, N, β, γ) for malaria or dengue yields four identifiable combinations from cumulative incidence: γ, μυ, **β/N**, and **(βυΛυ)/N**. The product of vector transmission rate and recruitment rate, divided by host population, forms a single identifiable combination. With known initial conditions, all parameters are identifiable.

**Ebola transmission model** with hospital and funeral transmission has **9 parameters** (βI, βH, βD, k, α, δI, δH, γH, γI). The identifiability depends critically on data types. Observing only new infections yields no individually identifiable parameters. Adding hospitalizations identifies α and **γI+δI**. Adding deaths identifies 7 combinations including **βD/βH** and **βI/βH** (transmission rate ratios). The complex combination **(αβH + βDδI + βIδH + βIγH)/βI** is identifiable, showing how multiple parameters can combine in non-obvious ways.

**COVID-19 model with pre-symptomatic transmission** uses **7 parameters** (βρ, βI, k, kρ, γ, γρ, δ). From case and death data, γ, δ, and βI are identifiable along with two constraint equations: **7βρ + 2γρ = constant** and **kρ + γρ = constant**. This model exhibits two solution sets, requiring k to be fixed for unique identification. This demonstrates local versus global identifiability—multiple parameter sets can fit data equally well.

## Age-structured PDE models extend to infinite dimensions

Renardy & Eisenberg (2022, *Journal of Mathematical Biology* 84(1):11) extended identifiability analysis to partial differential equations with age structure. Their **SEI model** with **7-9 parameters** (β, c, μS, μE, μI, γ, k, plus age-dependent coefficients) shows how spatial or age structure affects identifiability.

With frequency-dependent transmission, the identifiable parameters are β, γ, k, μS, μI, and the combination **βc/N** (transmission rate scaled by contact rate and population). Individual values of contact rate c and population N are unidentifiable, only their ratio with transmission. Adding immigration allows separation: β, γ, k, μS, μI, c, and m (immigration rate) all become individually identifiable.

With age-dependent death rates μ(a) = d₀exp(d₁a), both **d₀ and d₁ are individually identifiable** from age-distributed data. For polynomial death rates μ(a) = Σaᵢaⁱ, all coefficients are identifiable. This demonstrates that structural assumptions about functional forms directly determine identifiability of shape parameters. Code is available at https://github.com/epimath/age-pde-identifiability.

## Phenomenological growth models for outbreak forecasting

Chowell et al. (2025, *Infectious Disease Modelling*, PMC12031297) analyzed the GrowthPredict toolbox models used for epidemic forecasting. These simple models have high practical importance despite being phenomenological rather than mechanistic.

**Generalized logistic model (GLM)** has **4 parameters** (r, p, K, C0) representing growth rate, deceleration exponent, carrying capacity, and initial conditions. All parameters are structurally identifiable from incidence observations when properly reformulated. The growth rate r, carrying capacity K, and exponent p can be expressed as combinations of input-output equation coefficients derived through differential algebra. These models are implemented in the GrowthPredict MATLAB toolbox.

The **generalized growth model (GGM)** with **3 parameters** (r, p, C0) omits the carrying capacity, suitable for early outbreak phases. The **Gompertz model** with **3 parameters** and the **Richards model** with **4 parameters** complete the suite. All are structurally identifiable but may exhibit practical identifiability issues depending on data quality and outbreak phase.

## Tumor growth models with parameter correlation challenges

Benzekry et al. (2014, *PLOS Computational Biology*) analyzed classical tumor growth models using Lewis lung carcinoma and breast cancer xenograft data (20 and 34 mice respectively).

**Gompertz model** has **3 parameters** (a, β, K): initial proliferation rate, decay rate, and carrying capacity. The model equation dV/dt = -β×V×ln(V/K) or alternatively V(t) = exp(a/β × (1-e⁻ᵝᵗ)) shows both parameters are structurally identifiable with low standard errors. However, α and β exhibit correlation R²=0.968, indicating practical identifiability problems despite structural identifiability. Reparameterization using **k = α/β** (reduced Gompertz model) with β as a mixed effect improved identifiability, showing that parameter transformations guided by correlation analysis can enhance practical identifiability.

**Power law model** dV/dt = a×Vᵞ has **2 parameters** (a, γ) that are both well-identified. The exponent γ represents the fractional Hausdorff dimension of proliferative tissue. For LLC data, γ significantly differed from 2/3 or 1 in 14/20 mice, suggesting fractal vasculature drives proliferation. This small model provides good identifiability and biological interpretability.

**Generalized logistic model** dV/dt = aV(1-(V/K)ᵛ) with **3 parameters** (a, K, ν) showed poor identifiability despite best fit. High flexibility yielded high standard errors, making this model not recommended despite descriptive power. This demonstrates the identifiability-flexibility tradeoff inherent in model selection.

## Chemical engineering and reaction networks

Grunberg & Del Vecchio (2023, *SIAM Journal on Applied Dynamical Systems* 22(3), arXiv:2109.09943) analyzed chemical reaction networks using stationary distributions under the Linear Noise Approximation. They found that not all reaction rate constants are uniquely identifiable even with complete concentration dynamics. Confoundable networks exist where different network structures produce identical dynamics, representing a fundamental limitation on identifiability from stationary data alone. The method uses Hilbert's Nullstellensatz with symbolic computation to identify which parameter combinations are identifiable, implemented in Mathematica/DAISY.

**Batch reactor models** for chemical engineering use **4-10 parameters** depending on reaction network complexity. The Van de Vusse network example with 5 chemical species and 4 elementary reactions demonstrates that network structure identification plus rate constant estimation can be achieved from concentration trajectories using evolutionary strategies combined with qualitative trend analysis.

## Ecological and population dynamics models

**Generalized Lotka-Volterra model** for microbial communities (Royal Society Open Science, 2021, 8(3):201378) with N species has 2N growth rates and N² interaction coefficients. When only relative abundances are observable (as with DNA sequencing), absolute abundance data are required for unique parameter identification. However, **relative interaction strengths** are identifiable from relative abundance data alone, providing useful ecological information despite incomplete identifiability. This framework extends systematically to N-species communities.

Simpson et al. (2022, *Journal of Theoretical Biology* 535:110998) analyzed sigmoid population growth models. The **logistic model** with **2-3 parameters** has all parameters identifiable. The **Gompertz and Richards models** exhibit practical non-identifiability despite structural identifiability. Profile likelihood analysis revealed that confidence intervals vary dramatically across parameter space, with some parameter combinations having essentially infinite uncertainty. This demonstrates that structural identifiability guarantees existence but not practical computability of parameter estimates.

## Software tools and their example model libraries

**DAISY** (Differential Algebra for Identifiability of SYstems) by Bellu et al. (2007, *Computer Methods and Programs in Biomedicine* 88:52-61) pioneered computational identifiability analysis. Available at http://www.dei.unipd.it/~pia/, it uses differential algebra with characteristic sets in REDUCE. Example models include: (1) tumor model with antibody treatment, **5 parameters**, all globally identifiable; (2) macrophage mannose receptor pharmacokinetics, **6 unknown parameters** plus unknown initial condition, all globally identifiable; (3) 20-state unidirectional chain with Michaelis-Menten kinetics, **22 parameters**, all globally identifiable but requiring 150 minutes computation time demonstrating scaling limitations.

**STRIKE-GOLDD** by Villaverde et al. (2016) at https://github.com/afvillaverde/strike-goldd uses Lie derivatives and observability decomposition in MATLAB. It handles non-rational models (Hill functions, etc.) that previous methods struggled with. The seven detailed examples include Goodwin oscillator (**8 parameters**, 4 identifiable), MAPK cascade (**14 parameters**, identifiability depends on measured states), NF-κB (**29 parameters**, 25 identifiable), JAK/STAT (**23 parameters**, identifiable after fixing 1), and the large CHO metabolism model (**117 parameters** reduced to 111 identifiable).

**GenSSI** (Generating Series for Structural Identifiability) by Chiş et al. (2011, *Bioinformatics* 27(18):2610-2611) at https://github.com/genssi-developer/GenSSI uses identifiability tableaus showing which parameters depend on which Lie derivatives. GenSSI 2.0 added SBML import. The visual tableau representation helps users systematically solve for parameters from generating series coefficients.

**SIAN** (Structural Identifiability ANalyser) by Hong et al. (2019, *Bioinformatics* 35(16):2873-2874) at https://github.com/pogudingleb/SIAN provides Maple, Julia, and web app interfaces. Benchmark models include 2-4 compartment models, HIV models, SIRSForced epidemic model, and NF-κB. The web interface at https://maple.cloud/app/6509768948056064 makes analysis accessible without programming expertise.

**COMBOS** by Meshkat et al. (2014) at http://biocyb1.cs.ucla.edu/combos provides explicit symbolic expressions for identifiable parameter combinations via web interface. Built-in examples show 2-4 compartment models with clear mathematical expressions like "k₁₂·k₂₁ is uniquely identifiable" and "k₀₁ + k₂₁ is uniquely identifiable."

**StructuralIdentifiability.jl** is a modern Julia package using projections with differential elimination, providing high-performance computation for contemporary research.

## Common patterns in identifiable parameter combinations

Analysis across all models reveals recurring mathematical structures for identifiable combinations from non-identifiable models:

**Product combinations** appear most frequently in compartmental and signaling models: k₁₂·k₂₁ (exchange rates), β·p (viral dynamics), θ₁₇·θ₂₂ (JAK/STAT), Rtot·ke (receptors times transduction), βυΛυ/N (vector transmission product scaled by population).

**Sum combinations** appear in rate processes: k₀₁ + k₂₁ (competing outflows), k₀₂ + k₁₂ (turnover rates), γ+δ (recovery plus death), e₅₄ + e₅₅ (elasticity coefficients in lin-log kinetics).

**Ratio combinations** appear when scales are unidentifiable: β/N (per-capita rates), Rtot/RC₅₀ (transducer ratio), βD/βH (relative transmission), k = α/β (Gompertz ratio parameter), βc/N (contact-scaled transmission).

**Complex combinations** emerge in detailed mechanistic models: (αβH + βDδI + βIδH + βIγH)/βI from Ebola transmission, k₁₁k₂₂k₃₃ - k₁₂k₂₁k₃₃ - k₁₁k₂₃k₃₂ + k₁₃k₂₁k₃₂ + k₁₂k₂₃k₃₁ - k₁₃k₂₂k₃₁ from three-compartment determinant expressions.

## Practical workflow recommendations

The surveyed literature suggests a systematic approach: (1) Perform structural identifiability analysis early using DAISY, STRIKE-GOLDD, or SIAN before collecting expensive experimental data. (2) For structurally non-identifiable models, examine the identifiable parameter combinations—these may be sufficient to answer biological questions. (3) If individual parameters are needed, identify which combinations are identifiable and design experiments to measure additional states or use prior knowledge to fix correlated parameters. (4) Perform practical identifiability analysis via profile likelihood or MCMC even for structurally identifiable models, as high correlations (>0.9) indicate estimation problems despite theoretical identifiability. (5) Consider model reduction or reparameterization using identifiable combinations as new parameters rather than attempting to estimate unidentifiable originals.

## Conclusion: comprehensive resources now available

This survey identified 40+ models with 5-30 parameters having explicitly documented identifiable parameter combinations. The benchmarking paper by Rey Barreiro & Villaverde (2023) provides 25 standardized test cases. The STRIKE-GOLDD examples (Villaverde et al., 2016) offer detailed biological pathway models with complete code. The epidemic modeling tutorial (Chowell et al., 2023) provides systematic coverage with explicit mathematical expressions and GitHub code. Major software tools provide worked examples: DAISY (5+ models), STRIKE-GOLDD (7 detailed cases), COMBOS (web interface with instant results), and SIAN (multiple benchmarks). Mathematical expressions range from simple products and ratios to complex multiparameter combinations derived from observability matrix kernels or differential algebra. Code repositories ensure reproducibility, with GitHub becoming standard for sharing implementations. Publication venues span *PLOS Computational Biology*, *Bioinformatics*, *Mathematical Biosciences*, and domain journals, with a publication surge 2014-2023 driven by new computational methods and COVID-19 applications. These resources provide both theoretical foundations and practical tools for addressing parameter identifiability in mathematical modeling across all application domains.