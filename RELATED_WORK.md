# Related Work — draft prose + BibTeX

For `writeup.pdf` ("Reading Between the Lines: Assessing LLM Comprehension of Scientific Code").
Compiled 2026-07-24.

**Accuracy note:** every entry below was checked against arXiv/ACL/proceedings listings.
Author lists written as `{Surname, First and others}` were abbreviated *deliberately* because
the full list was not verified field-by-field — expand and check those before submission.
Entries with complete lists were read off the paper page directly.

---

## 1. Drafted Related Work section

### Code comprehension beyond generation

Most evaluation of code LLMs targets generation: functional correctness on
self-contained problems \cite{du2023classeval, jain2024livecodebench}. A separate line
measures *comprehension* — whether a model can predict what code does rather than write
it. CRUXEval \cite{gu2024cruxeval} evaluates input and output prediction on simple Python
functions; CodeMind \cite{liu2024codemind} separates implicit from explicit execution
reasoning; CodeSense \cite{roy2025codesense} extends this to fine-grained semantic
reasoning over real repositories and concludes that missing code semantics fundamentally
limits current models. These benchmarks share an assumption we relax: that the meaning of
a program is exhausted by its input–output behavior. For a PDE solver, two programs with
identical types and shapes can encode entirely different physical systems, and a program
that runs to completion without error can still be physically wrong. We therefore evaluate
comprehension against *physical* referents — the governing equation, the numerical scheme,
the dominant physical process, and physical validity — none of which is recoverable from
I/O behavior alone.

### Surface cues versus program semantics

A consistent finding is that model judgments about code track lexical surface form more
than program semantics. ReCode \cite{wang2023recode} introduced the standard protocol:
apply semantics-preserving transformations to docstrings, identifiers, syntax, and
formatting, and measure the drop. Le et al. \cite{le2025names} show that removing
meaningful identifiers degrades summarization and execution prediction, arguing that
programs carry meaning through both structural semantics and human-interpretable naming.
Haroon et al. \cite{haroon2025fault} find that semantics-preserving mutations cause LLMs
to fail on faults they had previously localized in 78% of cases across 10 models.
CodeCrash \cite{lam2025codecrash} is closest to our comment conditions: injecting
misleading natural language into CRUXEval and LiveCodeBench problems degrades output
prediction by 23.2% on average across 17 LLMs, with chain-of-thought reducing but not
eliminating the effect (13.8%), and reasoning models exhibiting *reasoning collapse* —
rationalizing the misleading cue at 2–3× token cost. Most fundamentally, Thimmaiah et al.
\cite{thimmaiah2025priors} show with PLSemanticsBench that when formal operational
semantics are mutated so that familiar symbols mean unfamiliar things, accuracy falls
40–60 points and long-horizon conditioning on the supplied rules peaks at 35% — direct
evidence that models lean on pretrained lexical association rather than supplied semantics.
Our contribution is to run this design where the *referent is physical rather than
computational*: our corrupted comments describe a coherent but different physical system,
and our invalid variants remain syntactically valid and (with one documented exception)
executable, so the model cannot detect them by spotting a crash. Relatedly, code–comment
inconsistency is a long-standing software-engineering problem in its own right
\cite{panthaplackel2021jit}; our `CorrComm` condition can be read as a synthetic
inconsistency benchmark in the scientific-code domain.

### Internal representations: what models encode versus what they say

Our probing results speak to a literature on the gap between what a model encodes and
what it outputs. Azaria and Mitchell \cite{azaria2023internal} show that hidden states
separate true from false statements; Burns et al. \cite{burns2023discovering} recover
latent truth directions without supervision; Orgad et al. \cite{orgad2025know} show that
probes can select the correct answer in cases where the model's own generation shows no
preference for it. In the code domain, Ribeiro et al. \cite{ribeiro2026correctness}
extract a correctness direction by contrasting hidden states of correct and incorrect
solutions to the same task, and show it outperforms both log-likelihood ranking and
verbalized confidence across four LLMs; related work assesses generated-code correctness
from internal representations without execution \cite{le2025autoprobe}. Probing has also
been used to characterize what code models encode structurally: AST-Probe
\cite{lopez2022astprobe} recovers whole abstract syntax trees from hidden
representations and locates most syntactic information in middle layers, and subsequent
work separates syntactic from semantic capacity \cite{ma2024unveiling}. We extend this
line to physical rather than syntactic or correctness-based content: we ask whether PDE
class, numerical method, and physical validity are linearly decodable, and whether
decodability survives the lexical perturbations that degrade the model's stated answers.

### Execution as evidence

A parallel line grounds code understanding in execution rather than static reading.
InterCode \cite{yang2023intercode} frames interactive coding as a reinforcement-learning
environment with execution feedback as observation; debug-gym \cite{yuan2025debuggym}
gives agents interactive debugger access, raising SWE-bench Lite resolution from 10.7% to
30.2% for one model, on the argument that agents must actively query the semantic space
hidden behind static code. Execution evidence has also been internalized during training:
NExT \cite{ni2024next} trains on execution-aware rationales, SemCoder \cite{ding2024semcoder}
uses monologue-style execution reasoning, and Execution Tuning
\cite{armengol2025execute} trains directly on program traces. Self-debugging shows that
execution results improve repair \cite{chen2024selfdebug}, while broader evaluations
caution that self-correction is reliable only when the external signal is
\cite{huang2024cannot, kamoi2024survey}. Two recent results complicate the picture in a
way that directly motivates our design: LLM agents can defer to tool outputs almost
wholesale — matching raw tool predictions 97.6–99.2% of the time, with deference
*increasing* with model scale \cite{wang2026tool} — while embodied agents exhibit the
opposite failure, *belief inertia*, adhering to priors despite contradicting observations
\cite{wang2026inertia}. Our belief-revision experiment sits between these poles: we supply
execution summaries for code whose physical validity is known to us, and measure whether
revision is concentrated where the model's lexical priors are weakest.

### Reasoning traces and their faithfulness

Because our stronger models are reasoning models, the relationship between the visible
trace and the operative computation is a live confound. Chain-of-thought explanations can
be systematically influenced by biasing features the model never verbalizes
\cite{turpin2023say}, and faithfulness can be measured by how much perturbing the trace
changes the answer \cite{lanham2023measuring}. Recent evaluations of frontier and
open-weight reasoning models find persistent unfaithfulness \cite{chen2025reasoning,
arcuschin2025wild}, and Young \cite{young2026know} reports that in over half of
hint-influenced cases the thinking tokens contain hint-related keywords the visible answer
omits entirely. This bears directly on our interpretation: a model whose stated validity
judgment collapses under comment corruption may or may not have registered the conflict
internally, which is precisely the dissociation our probes are designed to detect. Work
that routes reasoning through code — PAL \cite{gao2023pal}, Program of Thoughts
\cite{chen2023pot}, Chain of Code \cite{li2024coc} — offers a complementary lever, since
emulating a solver's execution is a different operation from recognizing its lexical
signature.

### Why scientific code: the oracle problem

Our choice of testbed rests on a property that software engineering has long recognized in
scientific software: for many scientific programs there is no test oracle. Kanewala and
Bieman \cite{kanewala2014testing} survey testing of scientific software and identify the
absence of a mechanism to decide whether a computed outcome is correct as the field's
defining difficulty, motivating pseudo-oracles, comparison against analytical solutions,
and metamorphic testing \cite{chen2018metamorphic} in place of conventional assertions. The
consequence is a characteristic failure mode — code that executes cleanly and returns
plausible numbers that are wrong — which has been argued to compromise published results at
scale \cite{soergel2014rampant}, and which reappears in agentic settings: Rawat and Flek
\cite{rawat2026plausible} report that in astrophysical workflows the dominant failure of an
LLM agent system is not a crash but "silent incorrect computation — syntactically valid code
that produces plausible but inaccurate results," including physically inconsistent posteriors
generated without self-diagnosis.

This is what makes PDE solvers a sharper instrument than generic code for our question. In
execution-based comprehension benchmarks \cite{gu2024cruxeval, roy2025codesense}, program
meaning is *defined by* input–output behavior, so a model that predicts behavior correctly
has, by construction, understood the program. A PDE solver admits a referent outside its own
execution: the governing equation and the physics it is meant to reproduce. A single-line
edit — flipping a sign, dropping a diffusive term, zeroing a time derivative — can leave the
program syntactically valid, executable, and numerically bounded while changing the system it
simulates. Correct behavioral prediction is therefore not sufficient for correct
comprehension, and the wrongness has a known location and a known physical interpretation,
which lets us ask *where* in the code a judgment is grounded rather than only whether it is
right. Scientific code additionally carries several partially redundant evidence channels —
identifier names, comments, the discretized update equation, the numerical scheme, and the
simulated trajectory — that can be placed in controlled conflict, which is the basis of our
factorial design. We do not claim this property is unique to physics; it holds wherever
correctness is externally defined, and Thimmaiah et al. \cite{thimmaiah2025priors} obtain a
comparable dissociation by mutating formal operational semantics. Physical simulation offers
a version in which the alternative semantics is not synthetic but real, documented, and
plausibly known to the model.

### LLMs for scientific and PDE code

Scientific-code evaluation has concentrated on generation and agentic workflows: SciCode
\cite{tian2024scicode} decomposes scientist-curated problems across 16 subfields,
ScienceAgentBench \cite{chen2025scienceagentbench} extracts tasks from peer-reviewed
papers, and ResearchCodeBench \cite{hu2025researchcodebench} evaluates paper-to-code
implementation. In the PDE setting specifically, CodePDE \cite{li2026codepde} treats PDE
solving as code generation with self-refinement, LANG-PINN \cite{he2025langpinn} generates
physics-informed neural networks from language, and further systems automate solver
development and refinement \cite{pdeveloper2025, pdesharp2025}. Our dataset draws solvers
from public scientific-computing sources including PDEBench \cite{takamoto2022pdebench}.
Physics reasoning independent of code remains difficult for frontier models
\cite{qiu2025phybench}, which bounds what we should expect from validity judgments over
code. Closest to our interpretability question, Song et al. \cite{song2025physics} find
features correlated with physical quantities in residual-stream activations when models
are given trajectory data; we ask the complementary question for code, where meaning is
carried simultaneously by executable structure and by names and comments that can be made
to disagree with it.

---

## 2. Positioning notes (not for the paper)

**Differentiation you must state explicitly, or reviewers will call this a replication:**

| Prior work | What it already establishes | Your delta — say it in one sentence |
|---|---|---|
| CodeCrash (NeurIPS'25) | Misleading NL degrades code reasoning 23.2%; QwQ-32B reasoning collapse | Their referent is I/O behavior; yours is a *physical system*, unrecoverable from execution alone. Also: their corruption is adversarial noise, yours is a coherent description of a *different real PDE*. |
| Le et al. 2025 (already cited) | Name removal degrades summarization/execution prediction | You add the validity axis and the probe/behavior dissociation. |
| Haroon et al. 2025 | Semantics-preserving mutations break fault localization 78% | Strongest support for your `*_InValid` results — cite as convergent, not competing. |
| Thimmaiah et al. (ICML'26) | Models condition on lexical priors, not supplied formal semantics (40–60 pt drop) | Same thesis, formal-PL domain. Your version is physical semantics with no rule sheet supplied. **Read this one.** |
| Ribeiro et al. (ICSE'26) | Contrastive hidden-state correctness direction beats logprob + verbalized confidence | Predicts your probe>behavior result for `valid`. Cite as precedent and show your result is about *physical* validity, not functional correctness. |
| Orgad et al. (ICLR'25) | Probes recover answers the output doesn't express | The framing citation for your abstract's key claim. |

**Threats the literature already names — address each in one sentence:**

1. **Contamination.** Sources (Gilbert François, Barba's *12 Steps to Navier–Stokes*,
   JAX-CFD, PDEBench) are among the most-replicated physics code on GitHub. A 100%
   `pde_class` probe may read a memorized repo fingerprint. Mitigation: Min-K%/verbatim
   completion check + a small hand-written held-out solver set. Cite \cite{mirzadeh2025gsm},
   \cite{xu2025contamination}.
2. **MCQ option order.** \cite{pezeshkpour2024order, zheng2024selectors} — if options
   weren't permuted, this is reviewer question #1. Logprob confidence caveat:
   \cite{gonzalez2025confident}.
3. **Belief revision confound.** Revision under execution summaries is consistent with
   genuine updating, blind deference \cite{wang2026tool}, or sycophancy
   \cite{sharma2024sycophancy, laban2024flipflop}. **Wrong-summary control arm resolves it**
   and is cheap on Gemini-2.5-Flash × 128 rows.
4. **Thinking disabled.** All reported runs suppress thinking. Given CodeCrash's finding
   that CoT partially recovers but can rationalize, a thinking-ON arm on the `CorrComm`
   conditions is close to free and materially strengthens the paper.

---

## 3. BibTeX

```bibtex
% ─────────────────────────────────────────────────────────────
% Code comprehension / reasoning benchmarks
% ─────────────────────────────────────────────────────────────

@inproceedings{gu2024cruxeval,
  title     = {{CRUXEval}: A Benchmark for Code Reasoning, Understanding and Execution},
  author    = {Gu, Alex and Rozi{\`e}re, Baptiste and Leather, Hugh and
               Solar-Lezama, Armando and Synnaeve, Gabriel and Wang, Sida I.},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2401.03065},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2401.03065}
}

@article{liu2024codemind,
  title   = {{CodeMind}: Evaluating Large Language Models for Code Reasoning},
  author  = {Liu, Changshu and Zhang, Shizhuo Dylan and Ibrahimzada, Ali Reza and
             Jabbarvand, Reyhaneh},
  journal = {arXiv preprint arXiv:2402.09664},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.09664}
}

@article{roy2025codesense,
  title   = {{CodeSense}: A Real-World Benchmark and Dataset for Code Semantic Reasoning},
  author  = {Roy, Monoshi Kumar and Chen, Simin and Steenhoek, Benjamin and
             Peng, Jinjun and Kaiser, Gail and Ray, Baishakhi and Le, Wei},
  journal = {arXiv preprint arXiv:2506.00750},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.00750}
}

% ─────────────────────────────────────────────────────────────
% Surface cues vs. semantics / robustness
% ─────────────────────────────────────────────────────────────

@inproceedings{lam2025codecrash,
  title     = {{CodeCrash}: Exposing {LLM} Fragility to Misleading Natural Language
               in Code Reasoning},
  author    = {Lam, Man Ho and Wang, Chaozheng and Huang, Jen-tse and Lyu, Michael R.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2504.14119},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2504.14119}
}

@article{haroon2025fault,
  title   = {Assessing the Impact of Code Changes on the Fault Localizability of
             Large Language Models},
  author  = {Haroon, Sabaat and Khan, Ahmad Faraz and Humayun, Ahmad and Gill, Waris and
             Amjad, Abdul Haddi and Butt, Ali R. and Khan, Mohammad Taha and
             Gulzar, Muhammad Ali},
  journal = {arXiv preprint arXiv:2504.04372},
  year    = {2025},
  note    = {Earlier circulated as ``How Accurately Do Large Language Models Understand Code?''},
  url     = {https://arxiv.org/abs/2504.04372}
}

@inproceedings{wang2023recode,
  title     = {{ReCode}: Robustness Evaluation of Code Generation Models},
  author    = {Wang, Shiqi and Li, Zheng and Qian, Haifeng and Yang, Chenghao and
               Wang, Zijian and Shang, Mingyue and Kumar, Varun and Tan, Samson and
               Ray, Baishakhi and Bhatia, Parminder and Nallapati, Ramesh and
               Ramanathan, Murali Krishna and Roth, Dan and Xiang, Bing},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for
               Computational Linguistics (ACL), Volume 1: Long Papers},
  pages     = {13818--13843},
  year      = {2023},
  url       = {https://aclanthology.org/2023.acl-long.773/}
}

@inproceedings{thimmaiah2025priors,
  title     = {{LLMs} Lean on Priors, Not Programming Language Semantics},
  author    = {Thimmaiah, Aditya and Zhang, Jiyang and Srinivasa, Jayanth and
               Li, Junyi Jessy and Gligoric, Milos},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  eprint    = {2510.03415},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2510.03415}
}

@inproceedings{panthaplackel2021jit,
  title     = {Deep Just-In-Time Inconsistency Detection Between Comments and Source Code},
  author    = {Panthaplackel, Sheena and Li, Junyi Jessy and Gligoric, Milos and
               Mooney, Raymond J.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {1},
  pages     = {427--435},
  year      = {2021},
  url       = {https://arxiv.org/abs/2010.01625}
}

% ─────────────────────────────────────────────────────────────
% Internal representations / probing
% ─────────────────────────────────────────────────────────────

@inproceedings{azaria2023internal,
  title     = {The Internal State of an {LLM} Knows When It's Lying},
  author    = {Azaria, Amos and Mitchell, Tom},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages     = {967--976},
  year      = {2023},
  url       = {https://aclanthology.org/2023.findings-emnlp.68/}
}

@inproceedings{burns2023discovering,
  title     = {Discovering Latent Knowledge in Language Models Without Supervision},
  author    = {Burns, Collin and Ye, Haotian and Klein, Dan and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2212.03827}
}

@inproceedings{orgad2025know,
  title     = {{LLMs} Know More Than They Show: On the Intrinsic Representation of
               {LLM} Hallucinations},
  author    = {Orgad, Hadas and Toker, Michael and Gekhman, Zorik and Reichart, Roi and
               Szpektor, Idan and Kotek, Hadas and Belinkov, Yonatan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.02707},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2410.02707}
}

@inproceedings{ribeiro2026correctness,
  title     = {On {LLMs}' Internal Representation of Code Correctness},
  author    = {Ribeiro, Francisco and Spiess, Claudio and Devanbu, Prem and Nadi, Sarah},
  booktitle = {Proceedings of the 48th IEEE/ACM International Conference on
               Software Engineering (ICSE)},
  year      = {2026},
  eprint    = {2512.07404},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2512.07404}
}

@article{le2025autoprobe,
  title   = {Model-Agnostic Correctness Assessment for {LLM}-Generated Code via
             Dynamic Internal Representation Selection},
  author  = {{Anonymous} and others},
  journal = {arXiv preprint arXiv:2510.02934},
  year    = {2025},
  note    = {VERIFY AUTHORS before submission},
  url     = {https://arxiv.org/abs/2510.02934}
}

@inproceedings{lopez2022astprobe,
  title     = {{AST-Probe}: Recovering Abstract Syntax Trees from Hidden
               Representations of Pre-Trained Language Models},
  author    = {Hern{\'a}ndez L{\'o}pez, Jos{\'e} Antonio and Weyssow, Martin and
               S{\'a}nchez Cuadrado, Jes{\'u}s and Sahraoui, Houari},
  booktitle = {Proceedings of the 37th IEEE/ACM International Conference on
               Automated Software Engineering (ASE)},
  year      = {2022},
  doi       = {10.1145/3551349.3556900},
  url       = {https://arxiv.org/abs/2206.11719}
}

@article{ma2024unveiling,
  title   = {Unveiling Code Pre-Trained Models: Investigating Syntax and
             Semantics Capacities},
  author  = {Ma, Wei and others},
  journal = {ACM Transactions on Software Engineering and Methodology},
  year    = {2024},
  doi     = {10.1145/3664606},
  note    = {VERIFY AUTHORS before submission},
  url     = {https://dl.acm.org/doi/10.1145/3664606}
}

% ─────────────────────────────────────────────────────────────
% Execution as evidence / agentic
% ─────────────────────────────────────────────────────────────

@inproceedings{yang2023intercode,
  title     = {{InterCode}: Standardizing and Benchmarking Interactive Coding with
               Execution Feedback},
  author    = {Yang, John and Prabhakar, Akshara and Narasimhan, Karthik and Yao, Shunyu},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2023},
  eprint    = {2306.14898},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2306.14898}
}

@article{yuan2025debuggym,
  title   = {debug-gym: A Text-Based Environment for Interactive Debugging},
  author  = {Yuan, Xingdi and others},
  journal = {arXiv preprint arXiv:2503.21557},
  year    = {2025},
  note    = {VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2503.21557}
}

@inproceedings{ni2024next,
  title     = {{NExT}: Teaching Large Language Models to Reason about Code Execution},
  author    = {Ni, Ansong and Allamanis, Miltiadis and Cohan, Arman and Deng, Yinlin and
               Shi, Kensen and Sutton, Charles and Yin, Pengcheng},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2404.14662},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2404.14662}
}

@inproceedings{ding2024semcoder,
  title     = {{SemCoder}: Training Code Language Models with Comprehensive Semantics
               Reasoning},
  author    = {Ding, Yangruibo and Peng, Jinjun and Min, Marcus J. and Kaiser, Gail and
               Yang, Junfeng and Ray, Baishakhi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  eprint    = {2406.01006},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2406.01006}
}

@article{armengol2025execute,
  title   = {What I Cannot Execute, I Do Not Understand: Training and Evaluating
             {LLMs} on Program Execution Traces},
  author  = {Armengol-Estap{\'e}, Jordi and Carbonneaux, Quentin and Zhang, Tianjun and
             Markosyan, Aram H. and Seeker, Volker and Cummins, Chris and
             Kambadur, Melanie and O'Boyle, Michael F. P. and Wang, Sida and
             Synnaeve, Gabriel and Leather, Hugh James},
  journal = {arXiv preprint arXiv:2503.05703},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.05703}
}

@inproceedings{chen2024selfdebug,
  title     = {Teaching Large Language Models to Self-Debug},
  author    = {Chen, Xinyun and Lin, Maxwell and Sch{\"a}rli, Nathanael and Zhou, Denny},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2304.05128},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2304.05128}
}

@inproceedings{huang2024cannot,
  title     = {Large Language Models Cannot Self-Correct Reasoning Yet},
  author    = {Huang, Jie and Chen, Xinyun and Mishra, Swaroop and
               Zheng, Huaixiu Steven and Yu, Adams Wei and Song, Xinying and Zhou, Denny},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2310.01798},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2310.01798}
}

@article{kamoi2024survey,
  title   = {When Can {LLMs} Actually Correct Their Own Mistakes? A Critical Survey
             of Self-Correction of {LLMs}},
  author  = {Kamoi, Ryo and Zhang, Yusen and Zhang, Nan and Han, Jiawei and Zhang, Rui},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {12},
  year    = {2024},
  doi     = {10.1162/tacl_a_00713},
  url     = {https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177}
}

@article{wang2026tool,
  title   = {When the Tool Decides: {LLM} Agents Defer Blindly to Graph Neural Network
             Tools, and Stronger Backbones Defer More},
  author  = {Wang, Zhongyuan and Vemuri, Pratyusha},
  journal = {arXiv preprint arXiv:2606.14476},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.14476}
}

@article{wang2026inertia,
  title   = {Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention
             in Embodied Agents},
  author  = {Wang, Hanlin and Leong, Chak Tou and Wang, Jian and Li, Wenjie},
  journal = {arXiv preprint arXiv:2604.17252},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.17252}
}

@inproceedings{sharma2024sycophancy,
  title     = {Towards Understanding Sycophancy in Language Models},
  author    = {Sharma, Mrinank and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  note      = {VERIFY full author list before submission},
  url       = {https://arxiv.org/abs/2310.13548}
}

@article{laban2024flipflop,
  title   = {Are You Sure? Challenging {LLMs} Leads to Performance Drops in
             the {FlipFlop} Experiment},
  author  = {Laban, Philippe and Murakhovs'ka, Lidiya and Xiong, Caiming and Wu, Chien-Sheng},
  journal = {arXiv preprint arXiv:2311.08596},
  year    = {2024},
  url     = {https://arxiv.org/abs/2311.08596}
}

% ─────────────────────────────────────────────────────────────
% Chain-of-thought faithfulness
% ─────────────────────────────────────────────────────────────

@inproceedings{turpin2023say,
  title     = {Language Models Don't Always Say What They Think: Unfaithful
               Explanations in Chain-of-Thought Prompting},
  author    = {Turpin, Miles and Michael, Julian and Perez, Ethan and Bowman, Samuel R.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2305.04388},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2305.04388}
}

@article{lanham2023measuring,
  title   = {Measuring Faithfulness in Chain-of-Thought Reasoning},
  author  = {Lanham, Tamera and others},
  journal = {arXiv preprint arXiv:2307.13702},
  year    = {2023},
  note    = {VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2307.13702}
}

@article{chen2025reasoning,
  title   = {Reasoning Models Don't Always Say What They Think},
  author  = {Chen, Yanda and others},
  journal = {arXiv preprint arXiv:2505.05410},
  year    = {2025},
  note    = {Anthropic. VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2505.05410}
}

@article{arcuschin2025wild,
  title   = {Chain-of-Thought Reasoning In The Wild Is Not Always Faithful},
  author  = {Arcuschin, Iv{\'a}n and others},
  journal = {arXiv preprint arXiv:2503.08679},
  year    = {2025},
  note    = {VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2503.08679}
}

@article{young2026know,
  title   = {Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence
             Between Thinking Tokens and Answers in Open-Weight Reasoning Models},
  author  = {Young, Richard J.},
  journal = {arXiv preprint arXiv:2603.26410},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.26410}
}

@inproceedings{gao2023pal,
  title     = {{PAL}: Program-Aided Language Models},
  author    = {Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, Pengfei and
               Yang, Yiming and Callan, Jamie and Neubig, Graham},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2211.10435}
}

@article{chen2023pot,
  title   = {Program of Thoughts Prompting: Disentangling Computation from Reasoning
             for Numerical Reasoning Tasks},
  author  = {Chen, Wenhu and Ma, Xueguang and Wang, Xinyi and Cohen, William W.},
  journal = {Transactions on Machine Learning Research},
  year    = {2023},
  url     = {https://arxiv.org/abs/2211.12588}
}

@inproceedings{li2024coc,
  title     = {Chain of Code: Reasoning with a Language Model-Augmented Code Emulator},
  author    = {Li, Chengshu and Liang, Jacky and Zeng, Andy and Chen, Xinyun and
               Hausman, Karol and Sadigh, Dorsa and Levine, Sergey and Fei-Fei, Li and
               Xia, Fei and Ichter, Brian},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/li24ar.html}
}

% ─────────────────────────────────────────────────────────────
% Scientific / PDE code
% ─────────────────────────────────────────────────────────────

@inproceedings{tian2024scicode,
  title     = {{SciCode}: A Research Coding Benchmark Curated by Scientists},
  author    = {Tian, Minyang and Gao, Luyu and Zhang, Shizhuo Dylan and others},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2024},
  eprint    = {2407.13168},
  archivePrefix = {arXiv},
  note      = {VERIFY full author list before submission},
  url       = {https://arxiv.org/abs/2407.13168}
}

@inproceedings{chen2025scienceagentbench,
  title     = {{ScienceAgentBench}: Toward Rigorous Assessment of Language Agents
               for Data-Driven Scientific Discovery},
  author    = {Chen, Ziru and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.05080},
  archivePrefix = {arXiv},
  note      = {VERIFY full author list before submission},
  url       = {https://arxiv.org/abs/2410.05080}
}

@article{hu2025researchcodebench,
  title   = {{ResearchCodeBench}: Benchmarking {LLMs} on Implementing Novel Machine
             Learning Research Code},
  author  = {Hu, Tianyu and others},
  journal = {arXiv preprint arXiv:2506.02314},
  year    = {2025},
  note    = {VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2506.02314}
}

@inproceedings{takamoto2022pdebench,
  title     = {{PDEBench}: An Extensive Benchmark for Scientific Machine Learning},
  author    = {Takamoto, Makoto and Praditia, Timothy and Leiteritz, Raphael and
               MacKinlay, Dan and Alesiani, Francesco and Pfl{\"u}ger, Dirk and Niepert, Mathias},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://arxiv.org/abs/2210.07182}
}

@article{qiu2025phybench,
  title   = {{PHYBench}: Holistic Evaluation of Physical Perception and Reasoning
             in Large Language Models},
  author  = {{PHYBench Team} and others},
  journal = {arXiv preprint arXiv:2504.16074},
  year    = {2025},
  note    = {VERIFY full author list before submission},
  url     = {https://openreview.net/forum?id=brG8FPq1cf}
}

@article{pdeveloper2025,
  title   = {Automated Code Development for {PDE} Solvers Using Large Language Models},
  author  = {{Anonymous} and others},
  journal = {arXiv preprint arXiv:2509.25194},
  year    = {2025},
  note    = {VERIFY AUTHORS before submission},
  url     = {https://arxiv.org/abs/2509.25194}
}

@article{pdesharp2025,
  title   = {{PDE-SHARP}: {PDE} Solver Hybrids Through Analysis and Refinement Passes},
  author  = {{Anonymous} and others},
  journal = {arXiv preprint arXiv:2511.00183},
  year    = {2025},
  note    = {VERIFY AUTHORS before submission},
  url     = {https://arxiv.org/abs/2511.00183}
}

% ─────────────────────────────────────────────────────────────
% The oracle problem / silent errors in scientific software
% ─────────────────────────────────────────────────────────────

@article{kanewala2014testing,
  title   = {Testing Scientific Software: A Systematic Literature Review},
  author  = {Kanewala, Upulee and Bieman, James M.},
  journal = {Information and Software Technology},
  volume  = {56},
  number  = {10},
  pages   = {1219--1232},
  year    = {2014},
  doi     = {10.1016/j.infsof.2014.05.006}
}

@article{chen2018metamorphic,
  title   = {Metamorphic Testing: A Review of Challenges and Opportunities},
  author  = {Chen, Tsong Yueh and Kuo, Fei-Ching and Liu, Huai and Poon, Pak-Lok and
             Towey, Dave and Tse, T. H. and Zhou, Zhi Quan},
  journal = {ACM Computing Surveys},
  volume  = {51},
  number  = {1},
  pages   = {1--27},
  year    = {2018},
  doi     = {10.1145/3143561}
}

@article{soergel2014rampant,
  title   = {Rampant Software Errors May Undermine Scientific Results},
  author  = {Soergel, David A. W.},
  journal = {F1000Research},
  volume  = {3},
  pages   = {303},
  year    = {2014},
  doi     = {10.12688/f1000research.5930.2},
  url     = {https://f1000research.com/articles/3-303/v1}
}

@article{rawat2026plausible,
  title   = {Plausible but Wrong: A Case Study on Agentic Failures in
             Astrophysical Workflows},
  author  = {Rawat, Shivam and Flek, Lucie},
  journal = {arXiv preprint arXiv:2604.25345},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.25345}
}

% ─────────────────────────────────────────────────────────────
% Contamination / evaluation validity
% ─────────────────────────────────────────────────────────────

@article{mirzadeh2025gsm,
  title   = {{GSM-Symbolic}: Understanding the Limitations of Mathematical Reasoning
             in Large Language Models},
  author  = {Mirzadeh, Iman and Alizadeh, Keivan and Shahrokhi, Hooman and Tuzel, Oncel and
             Bengio, Samy and Farajtabar, Mehrdad},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2025},
  url     = {https://arxiv.org/abs/2410.05229}
}

@article{xu2025contamination,
  title   = {Benchmarking Large Language Models Under Data Contamination:
             A Survey from Static to Dynamic Evaluation},
  author  = {Xu, Simin and others},
  journal = {arXiv preprint arXiv:2502.17521},
  year    = {2025},
  note    = {VERIFY full author list before submission},
  url     = {https://arxiv.org/abs/2502.17521}
}

@inproceedings{pezeshkpour2024order,
  title     = {Large Language Models Sensitivity to The Order of Options in
               Multiple-Choice Questions},
  author    = {Pezeshkpour, Pouya and Hruschka, Estevam},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2024},
  year      = {2024},
  url       = {https://aclanthology.org/2024.findings-naacl.130/}
}

@inproceedings{zheng2024selectors,
  title     = {Large Language Models Are Not Robust Multiple Choice Selectors},
  author    = {Zheng, Chujie and Zhou, Hao and Meng, Fandong and Zhou, Jie and Huang, Minlie},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2309.03882}
}

@article{gonzalez2025confident,
  title   = {Multiple Choice Questions: Reasoning Makes Large Language Models More
             Self-Confident Even When They Are Wrong},
  author  = {{Anonymous} and others},
  journal = {arXiv preprint arXiv:2501.09775},
  year    = {2025},
  note    = {VERIFY AUTHORS before submission},
  url     = {https://arxiv.org/abs/2501.09775}
}

% ─────────────────────────────────────────────────────────────
% Already in the paper (kept for completeness)
% ─────────────────────────────────────────────────────────────

@article{le2025names,
  title   = {When Names Disappear: Revealing What {LLMs} Actually Understand About Code},
  author  = {Le, Cuong Chi and Pham, Minh V. and Van, Cao Duy and Phan, Hoang N. and
             Phan, Hoang N. and Nguyen, Tien N.},
  journal = {arXiv preprint arXiv:2510.03178},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.03178}
}

@article{song2025physics,
  title   = {Uncovering Emergent Physics Representations Learned In-Context by
             Large Language Models},
  author  = {Song, Yeongwoo and Bae, Jaeyong and Kim, Dong-Kyum and Jeong, Hawoong},
  journal = {arXiv preprint arXiv:2508.12448},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.12448}
}

@article{li2026codepde,
  title   = {{CodePDE}: An Inference Framework for {LLM}-Driven {PDE} Solver Generation},
  author  = {Li, Shanda and Marwah, Tanya and Shen, Junhong and Sun, Weiwei and
             Risteski, Andrej and Yang, Yiming and Talwalkar, Ameet},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://arxiv.org/abs/2505.08783}
}
```
