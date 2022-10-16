# DSAG
The official pytorch implementation of [**DSAG: A Scalable Deep Framework for Action-Conditioned Multi-Actor Full Body Motion Synthesis**](https://arxiv.org/abs/2110.11460), [WACV 2023](https://wacv2023.thecvf.com/home).
Please visit our [**webpage**](https://skeleton.iiit.ac.in/dsag) for more details.

<img src = "images/Intro_diagram.jpg" />


## Table of contents:
1. Description
1. How to use

## Description
We introduce DSAG, a controllable deep neural framework for action-conditioned generation of full body multi-actor variable duration actions. To compensate for incompletely detailed finger joints in existing large-scale datasets, we introduce full body dataset variants with detailed finger joints. To overcome  shortcomings in existing generative approaches, we introduce dedicated representations for encoding finger joints. We also introduce novel spatiotemporal transformation blocks with multi-head self attention and specialized temporal processing. The design choices enable generations for a large range in body joint counts (24 - 52), frame rates (13 - 50), global body movement (in-place, locomotion) and action categories (12 - 120), across multiple datasets (NTU-120, HumanAct12, UESTC, Human3.6M).  Our experimental results demonstrate DSAG's significant improvements over state-of-the-art, its suitability for action-conditioned generation at scale. Code and models will be released.

<img src = "images/Architecture_merged.jpg" />
