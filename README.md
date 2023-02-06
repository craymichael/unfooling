# Unfooling Perturbation-Based Post Hoc Explainers

Code for our paper, "[Unfooling Perturbation-Based Post Hoc Explainers](https://arxiv.org/abs/2205.14772)." Here, we 
provide a solution to the scaffolding-based adversarial attack on
perturbation-based explainers. We propose a novel conditional anomaly detection
algorithm to detect and defend against the attack.

**Abstract**:

> Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP. 

**Note**: We adapt the "fooling" portion of the code from [Fooling-LIME-SHAP](https://github.com/dylan-slack/Fooling-LIME-SHAP)
in this repository.

## Installation
All required packages are listed in `requirements.txt`. This can be installed
in a virtual environment using tools, such as `virtualenv` or `conda`.

Example of installation via `pip`:

```shell
pip install -r requirements.txt
```

## Run Instructions

## How to Cite This Work

([link](https://arxiv.org/abs/2205.14772))
Zachariah Carmichael and Walter J. Scheirer. "Unfooling Perturbation-Based Post Hoc Explainers."
_Proceedings of the AAAI Conference on Artificial Intelligence_, 2023.

**BibTeX**:

```text
@inproceedings{CarmichaelUnfooling2023,
  title     = {Unfooling Perturbation-Based Post Hoc Explainers},
  author    = {Carmichael, Zachariah and Scheirer, Walter J.},
  year      = 2023,
  booktitle = {Proceedings of the {AAAI} Conference on Artificial Intelligence},
  publisher = {{AAAI} Press},
}
```