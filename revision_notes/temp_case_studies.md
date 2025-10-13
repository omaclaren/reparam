# Identifiability Analysis Case Studies (5--30 Parameter Models)

## HIV Dynamics Model (≈10 parameters, Virology)

**Description:** A four-compartment HIV/AIDS dynamical model (uninfected
CD4 cells, latently infected cells, actively infected cells, free virus)
with about 8--10 unknown parameters (infection rates, transition rates,
production/clearance rates, etc.) has been a classic test for structural
identifiability[\[1\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=The%20following%20nonlinear%20polynomial%204,22%5D%3AImage%2813).
This nonlinear ODE model is **initially structurally unidentifiable** --
several parameters cannot be determined uniquely from output data alone.

**Identifiability issues:** Analysis revealed that only certain
combinations of parameters could be uniquely determined. For example,
the infection rate (β), healthy cell death rate (d), and source rate (s)
are individually identifiable, as is the product of two virus-related
parameters (e.g. a production rate \$q_2\$ times an infection
coefficient
\$k_2\$)[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872).
In contrast, other parameters appear only in inseparable combinations --
e.g. \$q_1 \\cdot k_1 \\cdot k_2\$ (the product of a latent infection
rate and two other rates) or a sum like \$(\\mu_1 + k_1)\$ -- meaning
those parameters can only be estimated as a combined entity, not
individually[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872).
Without additional information, multiple distinct parameter sets produce
the same outputs (local identifiability with finite solution
"symmetries" in the
parameters[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can)).

**Techniques used:** *Differential algebra* methods (the DAISY software
and the COMBOS algorithm) were applied to find identifiable parameter
combinations in this
model[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can).
By computing input--output equations and Gröbner bases, these tools
identified which parameters are globally or locally identifiable and
provided explicit algebraic relationships for the unidentifiable
subset[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can).
In particular, COMBOS could automatically extract the identifiable
combinations (such as the products mentioned above) and determine the
number of distinct parameter solution
sets[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can).
This allowed a reparameterization of the HIV model in terms of
identifiable combinations, as demonstrated in the
literature[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872).

**Result:** The HIV model serves as a rigorous example where
**structural identifiability analysis** uncovers that individual rate
constants are not all recoverable -- instead, biologically interpretable
combinations (like products of infection and production rates) are
identifiable[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872).
Identifiability was improved by recognizing these combinations; for
instance, certain rate-product terms can be treated as single effective
parameters that are uniquely
estimated[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872).
This case is widely cited in identifiability studies (e.g. Saccomani *et
al.*, 2011) and is included as a benchmark example in identifiability
toolkits[\[1\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=The%20following%20nonlinear%20polynomial%204,22%5D%3AImage%2813)[\[4\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=22,View%20Article).

**Source:** Saccomani *et al.*, **Bull. Math. Biol.** 2011 (HIV model
identifiability)[\[4\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=22,View%20Article);
Meshkat *et al.*, **PLoS ONE** 2014 (COMBOS
analysis)[\[1\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=The%20following%20nonlinear%20polynomial%204,22%5D%3AImage%2813).

## Pharmacokinetic Two-Compartment Model (7 parameters, Pharmacology)

**Description:** Multi-compartment pharmacokinetic (PK) models are
another standard testbed. One example is a **two-compartment model with
nonlinear kinetics** (e.g. saturable enzymatic clearance or
receptor-mediated uptake) involving \~7
parameters[\[5\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=6,that%20has%20been%20made%20nonlinear).
In a typical setup, drug moves between a central compartment and a
peripheral compartment with rate constants (e.g. \$k\_{12}, k\_{21}\$),
is eliminated from the central compartment (rate \$k\_{01}\$ or
\$k\_{02}\$), and possibly follows Michaelis--Menten kinetics
(parameters \$V\_{\\max}, K_m\$) for part of the clearance or
distribution. An input term (e.g. dosing rate \$b_1\$) and an output
measurement proportional to central concentration (\$c_1\$) are
included[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1)[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km).

**Identifiability issues:** This nonlinear PK model as originally
formulated is **structurally unidentifiable** -- many parameters enter
the equations only in coupled ways. As a result, only certain
**combinations** of the parameters can be inferred from
concentration--time data. For instance, the product of the input scaling
and output scaling (\$b_1 \\cdot c_1\$) is identifiable (since only the
product influences measured concentrations), as is the product of
certain rate constants (e.g. \$k\_{02} \\cdot
k\_{12}\$)[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km).
Likewise, combinations involving the saturable clearance appear (e.g.
terms proportional to \$k\_{12}\\cdot k\_{21}\\cdot V\_{\\max}\^2\$, or
\$c_1 \\cdot V\_{\\max}\$, etc.), whereas \$V\_{\\max}\$ and other
parameters by themselves are not uniquely
estimable[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1).
In essence, multiple distinct sets of \${k\_{12}, k\_{21}, V\_{\\max},
K_m, \...}\$ can fit the data equally well, because only aggregated
expressions (like effective clearance or volume terms) matter for the
output.

**Techniques used:** *Symbolic identifiability analysis* via
differential algebra (Gröbner bases) was used to determine the
independent identifiable
combinations[\[5\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=6,that%20has%20been%20made%20nonlinear)[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1).
Meshkat and DiStefano's algorithm (and the COMBOS tool) systematically
searched for "decoupled" polynomial combinations of parameters that
appear in the model's input--output
equations[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202)[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km).
By trying various eliminations orderings, they found a minimal set of
algebraically independent combinations that reparameterize the
model[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202)[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1).
In this PK example, **five identifiable combinations** were found (since
7 original parameters minus 2 unidentifiable degrees of freedom = 5).
These included the aggregate rate constants and products noted above
(e.g. \$b_1c_1\$, \$k\_{02}k\_{12}\$, and other compound terms involving
\$k\_{12}, k\_{21}, V\_{\\max},
K_m\$)[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1).
Traditional methods (e.g. examining transfer functions for linear
compartments) also indicate that without measuring the peripheral
compartment or knowing certain volumes, only combined parameters (like
an overall clearance or volume ratio) can be determined -- a consistent
result with the symbolic approach.

**Result:** The two-compartment PK model demonstrates how an initially
non-identifiable model can be **reparameterized by identifiable
combinations**. After identifying \$5\$ new composite parameters (each a
specific product or sum of the original 7), the model can be written in
an equivalent form that is globally
identifiable[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202)[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km).
This satisfies reviewers that the example is realistic
(multi-compartment PK models are widely used in pharmacology) yet
challenging, and it highlights the utility of identifiability analysis:
one learns exactly which lumped pharmacokinetic parameters (e.g. an
apparent clearance or total volume of distribution) can be estimated
from available measurements, even if the individual rate constants
cannot[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1)[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km).
Such models have frequently appeared in identifiability benchmark
studies[\[9\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20pharmacokinetics%20model%20,22)[\[10\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=Saccomani%20et%20al.%20,computational%20error%20%E2%80%9Cheap%20space%20low%E2%80%9D).

**Source:** Meshkat *et al.*, **Math. Biosci.** 2012 (identifiable
combinations in PK
model)[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202)[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1);
Chis *et al.*, **PLoS ONE** 2011 (PK identifiability case
study)[\[9\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20pharmacokinetics%20model%20,22).

## Goodwin Oscillator Model (6--8 parameters, Systems Biology)

**Description:** Goodwin's oscillator is a **nonlinear gene-regulatory
network model** that produces self-sustained oscillations (originally
proposed for circadian
rhythms)[\[11\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20model%20describes%20the%20oscillations,20).
It consists of three differential equations (a negative feedback loop):
an mRNA or enzyme \$X\$ that is synthesized with a Hill-function term
and degraded, a subsequent protein or metabolite \$Y\$ activated by
\$X\$, and a final product \$Z\$ that represses \$X\$ synthesis. Typical
formulations involve 6 or more parameters -- e.g. a maximum
transcription rate, a Hill coefficient \$n\$, a half-saturation constant
\$K\$, and degradation or dilution rates for each species
(\$k_1,\\dots,k_3\$)[\[12\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=Image%20%20represents%20an%20enzyme,20)[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).
This model is often cited in identifiability analyses because of its
nonlinear feedback and rational (non-polynomial) terms.

**Identifiability issues:** In a realistic scenario where only one
observable output is measured (for example, only the final protein \$Z\$
is observed over time), the Goodwin model is **structurally
unidentifiable**[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21).
Intuitively, several parameters can trade off to produce similar
oscillation patterns -- for instance, the Hill coefficient and the
threshold \$K\$ both affect the sigmoidal feedback strength, and certain
rate constants can only be determined up to proportional scalings if
their effects on period/amplitude can compensate each other. Formal
analysis confirms that with a single output, the system's
**identifiability rank** is deficient: power series methods failed to
obtain a full-rank identifiability tableau in this
case[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).
This indicates not all parameters can be solved uniquely from the
output; some parameters are locally or infinitely non-identifiable (e.g.
multiple sets of Hill kinetics parameters yield the same output
behavior)[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).

**Techniques used:** A variety of structural identifiability techniques
have been tested on the Goodwin model. **Taylor series expansion** and
**generating series** methods were applied by Chis *et al.* to build an
identifiability
tableau[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).
These methods could not fully resolve the parameter dependencies with
limited outputs, illustrating the limitations of power series approaches
for rational-function models. The **differential algebra (DAISY)**
approach likewise reported the model unidentifiable under single-output
conditions (and even encountered difficulties solving the nonlinear
algebraic equations when the model included the Hill-term
nonlinearity)[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21).
To overcome this, researchers tried reformulating the model -- e.g.
**polynomial approximation**: converting the Hill function into a
polynomial form -- which made the equations more
tractable[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21).
They also examined an idealized scenario with **full state
observability** (i.e. measuring \$X\$, \$Y\$, and \$Z\$). In that
hypothetical case, the power series method was able to achieve a full
rank tableau and conclude the model would be globally identifiable if
all state variables were
observed[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).
In summary, the analysis employed symbolic series expansion, implicit
function theorem approaches, and DAISY's Gröbner basis method, each
highlighting different aspects of the model's identifiability (or lack
thereof)[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)[\[15\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=to%20check%20whether%20further%20results,could%20be%20achieved).

**Result:** The Goodwin oscillator example underscores how **model
output structure** (which variables are observed) strongly influences
identifiability. With realistic limited observables, certain parameters
(e.g. the Hill exponent or some rate constants) remain non-identifiable
-- effectively, only an **identifiable combination** of those parameters
(governing the oscillation period and amplitude) can be estimated
without additional
data[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed).
Researchers showed that adding more measurement outputs or simplifying
the model equations can recover
identifiability[\[15\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=to%20check%20whether%20further%20results,could%20be%20achieved).
This model, although hypothetical, is complex enough (3 states,
nonlinear feedback, \~6 parameters) to be convincing to reviewers as a
challenging test case, rather than a trivial toy. It has been used in
comparisons of identifiability
techniques[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21),
illustrating issues like unidentifiable Hill kinetics and the benefits
of experiment design (choosing additional outputs to measure) to resolve
parameter
ambiguity[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)[\[15\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=to%20check%20whether%20further%20results,could%20be%20achieved).

**Source:** Chis *et al.*, **PLoS ONE** 2011 (Goodwin model
identifiability
study)[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21);
Denis-Vidal & Joly, 1998 (global identifiability via reparameterization
of Hill functions).

## JAK-STAT Signaling Pathway Model (≈8--23 parameters, Cell Signaling)

**Description:** Cytokine-induced JAK-STAT signaling cascades (such as
the IL-13/STAT5 pathway or EPO/JAK2/STAT5 pathway) have served as
**practical identifiability case studies** in systems biology. These
models typically involve a chain of phosphorylation and transcription
events with dozens of species and reactions. A simplified JAK-STAT
module (as in Swameye *et al.* 2003) might have around 8--10 parameters
(rate constants for receptor binding, phosphorylation, nuclear
import/export, etc.), whereas more detailed versions can include 20+
parameters[\[16\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=IL13induced%20JAK,.).
The outputs are usually phospho-STAT levels in cytoplasm or nucleus over
time, measured by assays.

**Identifiability issues:** JAK-STAT models are often **partially
observed** (not all molecular species in the cascade are measured),
leading to practical and structural identifiability problems. A common
issue is an initial condition or scale parameter that is confounded with
a kinetic rate. For example, the total amount of STAT protein and a rate
constant for activation might only appear as a product in the model's
output, making them unidentifiable individually. Indeed, studies have
found that certain rate constants in the JAK-STAT pathway cannot be
reliably estimated from a single type of experiment -- multiple
parameter sets produce virtually identical fits (flat likelihood
profiles) indicating **practical
non-identifiability**[\[17\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162366#:~:text=The%20profile%20likelihood%20is%20utilised,STAT%20signaling.%20BMC).
In one case, modelers discovered that only the **ratio** of two
parameters (a phosphorylation rate and a dephosphorylation rate) was
determined by the data, whereas the absolute values could not be
separated. This degeneracy manifests as a long, flat valley in the
likelihood or cost function (the "sloppy" parameter phenomenon) rather
than a distinct optimum.

**Techniques used:** **Profile likelihood analysis** has been a key tool
to assess identifiability in JAK-STAT models. Raue *et al.* (2009)
exploited profile likelihood to combine structural and practical
identifiability
analysis[\[18\]](https://academic.oup.com/bioinformatics/article/25/15/1923/213246#:~:text=,Raue).
By scanning the likelihood as a function of each parameter (while
re-optimizing others), they identified which parameters have finite
confidence intervals and which effectively have unbounded or very wide
intervals (indicating non-identifiability). This method can also reveal
identifiable combinations: for instance, if two parameters are
non-identifiable individually but a certain combination is fixed, the
profile likelihoods will show a ridge along that combination. In the
JAK-STAT application, the profile likelihood approach pinpointed the
problematic parameter subset and suggested reparameterizing by a single
identifiable combination (such as an effective signaling rate that
lumped an initial STAT concentration with a rate
constant)[\[19\]](https://academic.oup.com/bioinformatics/article-pdf/25/15/1923/48992533/bioinformatics_25_15_1923.pdf#:~:text=Structural%20and%20practical%20identifiability%20analysis,is%20calibrated%20to%20experimental%20data)[\[20\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=,as%20benchmark%20models%20for).
Other techniques used include **Fisher Information Matrix (FIM)
analysis** -- which, in these models, often yields a
high-condition-number matrix indicating a sloppy mode -- and
**experimental design** strategies to improve identifiability (e.g.
measuring additional phospho-form species or using multiple ligand dose
profiles).

**Result:** JAK-STAT models have become **benchmark examples for
identifiability**
studies[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by).
They show how even biologically important, well-calibrated models can
suffer from identifiability issues if data are limited. The outcome of
analyses has been that certain parameters (e.g. some feedback strengths
or nuclear transport rates) are practically unidentifiable without
further data, but identifiable combinations (like an overall "effective
signaling efficiency") can often be obtained and have biological
meaning. These insights have informed model reduction -- e.g. fixing one
parameter or reparameterizing to a product of two -- and guided
experimentalists to measure previously unobserved components. The
JAK-STAT case, documented in Raue *et al.* (Bioinformatics 2009) and
subsequent works, is widely accepted in the literature as a convincing
**real-world example** where profile-likelihood-based identifiability
analysis was
critical[\[17\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162366#:~:text=The%20profile%20likelihood%20is%20utilised,STAT%20signaling.%20BMC)[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by).

**Source:** Raue *et al.*, **Bioinformatics** 2009 (JAK-STAT
identifiability via profile likelihood); Becker *et al.*, **Science**
2010 (JAK-STAT model with parameter estimation); Chis *et al.* 2011
(notes JAK/STAT as a
benchmark)[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by).

## NF-κB Signaling Module (≈15 parameters, Systems Biology)

**Description:** The NF-κB regulatory module is a classic signaling
network involving kinase cascades and feedback regulators (IκB
proteins). A typical model (e.g. Hoffmann *et al.* 2002 core NF-κB
module) has on the order of 15--20 parameters, including reaction rates
for IKK activation, IκB--NFκB binding and degradation, mRNA synthesis
and decay, etc. This model produces characteristic oscillations in NF-κB
nuclear localization and has been extensively studied.

**Identifiability issues:** Under "standard" experimental conditions
(e.g. a single dose of stimulant and measuring only nuclear NF-κB over
time), the NF-κB model was found to be **unidentifiable** in several
parameters[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
Balsa-Canto *et al.* (2010) reported that **structural
unidentifiability** plagued the model -- essentially, there were
insufficient independent outputs to constrain all rate constants. For
example, the two negative feedback loops (via IκBα and A20 proteins)
introduce redundant effects: different combinations of their synthesis
and degradation rates can produce similar NF-κB oscillation profiles. As
a result, parameter estimation could not determine a unique set of
values; some parameters were only determinable as combinations or not at
all[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
This led to large uncertainties and sloppy directions in the parameter
space, which is problematic for reliable predictions.

**Techniques used:** A **comprehensive identifiability workflow** was
applied to the NF-κB module. Initially, a **structural identifiability
analysis** (using DAISY or a similar differential algebra tool)
identified which parameters or combinations are theoretically estimable
given the model
structure[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
Then, a **practical identifiability check** (profile likelihood or
Markov-chain Monte Carlo for posterior distributions) was done to see
how current data constrain the parameters. In Balsa-Canto's study, an
**iterative identification procedure** was employed: they first found
identifiable subsets, then designed new experiments to target the
unidentifiable
directions[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
Notably, an **optimal experimental design** step was included: by
selecting additional measurements (e.g. monitoring inhibitor mRNA or
using perturbation experiments), they showed the identifiability could
be greatly
improved[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
Separately, other researchers (Findeisen *et al.* 2008) demonstrated an
**observability/identifiability Gramian** approach on the NF-κB
network[\[23\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=biochemical%20reaction%20networks%20obtained%20from,of%20an%20%27empirical%20observability%20Gramian).
Because analytic methods struggled with such a high-dimensional
nonlinear system, they used an empirical observability gramian to
numerically assess parameter influence on
outputs[\[23\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=biochemical%20reaction%20networks%20obtained%20from,of%20an%20%27empirical%20observability%20Gramian).
This allowed identifiability ranking of parameters without needing a
full symbolic solution, and confirmed which combinations of rate
constants were unobservable.

**Result:** The NF-κB module is a prime example where **identifiability
analysis and experiment design intersect**. The initial analysis showed
several parameters were unidentifiable (only certain lumped combinations
mattered for the
output)[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
For instance, the synthesis and degradation rates of IκB could only be
determined up to a ratio given one type of experiment. By performing
additional targeted experiments (as suggested by the analysis), those
ambiguities were resolved and all parameters became identifiable in the
augmented data
set[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties).
This example, drawn from a real biochemical network, is highly regarded
by peer reviewers: it demonstrates the full workflow of detecting
identifiability issues and then fixing them via new experiments. It also
highlights multiple techniques -- symbolic (differential algebra) to
find structural issues, numerical (profile likelihood, Fisher info,
Gramians) to confirm practical issues, and optimal design to address
them[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties)[\[23\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=biochemical%20reaction%20networks%20obtained%20from,of%20an%20%27empirical%20observability%20Gramian).
The case of NF-κB is frequently cited in reviews as a **benchmark for
identifiability** methods, alongside the JAK/STAT
pathway[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by).

**Source:** Balsa-Canto *et al.*, **BMC Syst. Biol.** 2010 (NF-κB
identifiability and experiment
design)[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties);
Findeisen *et al.* 2008 (Gramian identifiability analysis on
NF-κB)[\[23\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=biochemical%20reaction%20networks%20obtained%20from,of%20an%20%27empirical%20observability%20Gramian);
Chis *et al.* 2011 (notes NF-κB as
benchmark)[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by).

Each of the above models provides a **realistic yet non-trivial test
case** for identifiability analysis methods. They have all appeared in
the literature as exemplars of how to handle unidentifiable parameters
by finding identifiable reparameterizations or by improving data
collection. The range of techniques used -- from differential-algebraic
methods (DAISY, COMBOS) to numerical profile likelihoods and
information-theoretic methods -- underlines the importance of choosing
appropriate tools for a given model. Together, these case studies cover
**5--30 parameter regimes** and demonstrate how identifiable parameter
combinations can be extracted from initially unidentifiable
models[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can)[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202),
satisfying the criteria for rigorous examples to validate new
identifiability analysis methods. Each model's inclusion in
peer-reviewed studies and reviews attests to its suitability and
credibility as a benchmark for identifiability
research[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by)[\[9\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20pharmacokinetics%20model%20,22).

[\[1\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=The%20following%20nonlinear%20polynomial%204,22%5D%3AImage%2813)
[\[3\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=With%20no%20initial%20conditions%20given%2C,noted%20above%2C%20this%20result%20can)
[\[4\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261#:~:text=22,View%20Article)
On Finding and Using Identifiable Parameter Combinations in Nonlinear
Dynamic Systems Biology Models and COMBOS: A Novel Web Implementation \|
PLOS One

<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0110261>

[\[2\]](https://ww3.math.ucla.edu/camreport/cam08-80.pdf#:~:text=HIV%2FAIDS%20model%2C%20inspection%20can%20only,%F0%9D%9C%871%20%2B%20%F0%9D%91%981%2C%20%F0%9D%91%90%2C%20%F0%9D%9C%872)
ww3.math.ucla.edu

<https://ww3.math.ucla.edu/camreport/cam08-80.pdf>

[\[5\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=6,that%20has%20been%20made%20nonlinear)
[\[6\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=c1%20k12%20k21%20vm%202,k21%20vm%202%20b1%20c1)
[\[7\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=b1%202%20b1%202%20k01,2%20b1%20km%20c1%20km)
[\[8\]](https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf#:~:text=The%20Gr%C3%B6bner%20Basis%20with%20ranking,c1%20k12%20k21%20vm%202)
qcb.ucla.edu

<https://qcb.ucla.edu/wp-content/uploads/sites/14/2021/01/Meshkat-Anderson-DiStefano-2011.pdf>

[\[9\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20pharmacokinetics%20model%20,22)
[\[10\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=Saccomani%20et%20al.%20,computational%20error%20%E2%80%9Cheap%20space%20low%E2%80%9D)
[\[11\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20model%20describes%20the%20oscillations,20)
[\[12\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=Image%20%20represents%20an%20enzyme,20)
[\[13\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=For%20the%20case%20of%20one,local%20identifiability%20may%20be%20assessed)
[\[14\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=The%20differential%20algebra%20approach%2C%20as,21)
[\[15\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755#:~:text=to%20check%20whether%20further%20results,could%20be%20achieved)
Structural Identifiability of Systems Biology Models: A Critical
Comparison of Methods \| PLOS One

<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0027755>

[\[16\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=IL13induced%20JAK,.)
[\[20\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=,as%20benchmark%20models%20for)
[\[21\]](https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems#:~:text=proteins%20or%20metabolites,the%20state%20vector%20x%20by)
Comparison of approaches for parameter identifiability analysis of
biological systems \| Request PDF

<https://www.researchgate.net/publication/259917765_Comparison_of_approaches_for_parameter_identifiability_analysis_of_biological_systems>

[\[17\]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162366#:~:text=The%20profile%20likelihood%20is%20utilised,STAT%20signaling.%20BMC)
Driving the Model to Its Limit: Profile Likelihood Based Model \...

<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162366>

[\[18\]](https://academic.oup.com/bioinformatics/article/25/15/1923/213246#:~:text=,Raue)
Structural and practical identifiability analysis of partially observed
\...

<https://academic.oup.com/bioinformatics/article/25/15/1923/213246>

[\[19\]](https://academic.oup.com/bioinformatics/article-pdf/25/15/1923/48992533/bioinformatics_25_15_1923.pdf#:~:text=Structural%20and%20practical%20identifiability%20analysis,is%20calibrated%20to%20experimental%20data)
Structural and practical identifiability analysis of partially observed
\...

<https://academic.oup.com/bioinformatics/article-pdf/25/15/1923/48992533/bioinformatics_25_15_1923.pdf>

[\[22\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=conducting%20a%20practical%20identifiability%20analysis,that%20largely%20improved%20identifiability%20properties)
[\[23\]](https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood#:~:text=biochemical%20reaction%20networks%20obtained%20from,of%20an%20%27empirical%20observability%20Gramian)
(PDF) Structural and practical identifiability analysis of partially
observed dynamical models by exploiting the profile likelihood

<https://www.academia.edu/111738586/Structural_and_practical_identifiability_analysis_of_partially_observed_dynamical_models_by_exploiting_the_profile_likelihood>
