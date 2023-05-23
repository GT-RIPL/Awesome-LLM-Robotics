# Awesome-LLM-Robotics [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo contains a curative list of **papers using Large Language/Multi-Modal Models for Robotics/RL**. Template from [awesome-Implicit-NeRF-Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics) <br>

#### Please feel free to send me [pull requests](https://github.com/GT-RIPL/Awesome-LLM-Robotics/blob/main/how-to-PR.md) or [email](mailto:zkira-changetoat-gatech--changetodot-changetoedu) to add papers! <br>

If you find this repository useful, please consider [citing](#citation) and STARing this list. Feel free to share this list with others!

---
## Overview

  - [Reasoning](#reasoning)
  - [Planning](#planning)
  - [Manipulation](#manipulation)
  - [Instructions and Navigation](#instructions-and-navigation)
  - [Simulation Frameworks](#simulation-frameworks)
  - [Citation](#citation)

---
## Reasoning

 * **Instruct2Act**: "Mapping Multi-modality Instructions to Robotic Actions with Large Language Model", *arXiv, May 2023*. [[Paper](https://arxiv.org/pdf/2305.11176.pdf)]  [[Pytorch Code](https://github.com/OpenGVLab/Instruct2Act)]

 * **TidyBot**: "Personalized Robot Assistance with Large Language Models",  *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.05658)] [[Pytorch Code](https://github.com/jimmyyhwu/tidybot/tree/main/robot)] [[Website](https://tidybot.cs.princeton.edu/)]

 * **PaLM-E**: "PaLM-E: An Embodied Multimodal Language Model", *arXiv, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.03378)] [[Webpage](https://palm-e.github.io/)]
 
 * **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *arXiv, Dec 2022*. [[Paper](https://arxiv.org/abs/2212.06817)]  [[GitHub](https://github.com/google-research/robotics_transformer)] [[Website](https://robotics-transformer.github.io/)]

 * **ProgPrompt**: "Generating Situated Robot Task Plans using Large Language Models", arXiv, Sept 2022. [[Paper](https://arxiv.org/abs/2209.11302)]  [[Github](https://github.com/progprompt/progprompt)] [[Website](https://progprompt.github.io/)]

 * **Code-As-Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sept 2022*. [[Paper](https://arxiv.org/abs/2209.07753)]  [[Colab](https://github.com/google-research/google-research/tree/master/code_as_policies)] [[Website](https://code-as-policies.github.io/)]

 * **Say-Can**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.01691)]  [[Colab](https://say-can.github.io/#open-source)] [[Website](https://say-can.github.io/)]

 * **Socratic**: "Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.00598)] [[Pytorch Code](https://socraticmodels.github.io/#code)] [[Website](https://socraticmodels.github.io/)]

 * **PIGLeT**: "PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World", *ACL, Jun 2021*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](http://github.com/rowanz/piglet)] [[Website](https://rowanzellers.com/piglet/)]


---
## Planning

 * **LLM+P**:"LLM+P: Empowering Large Language Models with Optimal Planning Proficiency", *arXiv, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.11477)] [[Code](https://github.com/Cranial-XIX/llm-pddl)]

 * "Foundation Models for Decision Making: Problems, Methods, and Opportunities", *arXiv, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.04129)]

 * **PromptCraft**: "ChatGPT for Robotics: Design Principles and Model Abilities", *Blog, Feb 2023*, [[Paper](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)] [[Website](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)]

 * **Text2Motion**: "Text2Motion: From Natural Language Instructions to Feasible Plans", *arXiV, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.12153)] [[Website](https://sites.google.com/stanford.edu/text2motion)]

 * **ChatGPT-Prompts**: "ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application", *arXiv, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.03893?s=03)] [[Code/Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts)]

 * **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Pytorch Code](https://github.com/blazejosinski/lm_nav)] [[Website](https://sites.google.com/view/lmnav)]

 * **InnerMonlogue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Website](https://innermonologue.github.io/)]

 * **Housekeep**: "Housekeep: Tidying Virtual Households using Commonsense Reasoning", *arXiv, May 2022*. [[Paper](https://arxiv.org/abs/2205.10712)] [[Pytorch Code](https://github.com/yashkant/housekeep)] [[Website](https://yashkant.github.io/housekeep/)]

 * **LID**: "Pre-Trained Language Models for Interactive Decision-Making", *arXiv, Feb 2022*. [[Paper](https://arxiv.org/abs/2202.01771)] [[Pytorch Code](https://github.com/ShuangLI59/Language-Model-Pre-training-Improves-Generalization-in-Policy-Learning)] [[Website](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)]

 * **ZSP**: "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents", *ICML, Jan 2022*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](https://github.com/huangwl18/language-planner)] [[Website](https://wenlong.page/language-planner/)]

---
## Manipulation

 * **ProgramPort**:"Programmatically Grounded, Compositionally Generalizable Robotic Manipulation", "ICLR, Apr 2023", [[Paper](https://arxiv.org/abs/2304.13826)] [[Website] (https://progport.github.io/)]
 
 * **CoTPC**:"Chain-of-Thought Predictive Control", "arXiv, Apr 2023", [[Paper](https://arxiv.org/abs/2304.00776)] [[Code](https://github.com/SeanJia/CoTPC)]

 * **DIAL**:"Robotic Skill Acquistion via Instruction Augmentation with Vision-Language Models", "arXiv, Nov 2022", [[Paper](https://arxiv.org/abs/2211.11736)] [[Website](https://instructionaugmentation.github.io/)]

 * **CLIP-Fields**:"CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory", "arXiv, Oct 2022", [[Paper](https://arxiv.org/abs/2210.05663)] [[PyTorch Code](https://github.com/notmahi/clip-fields)] [[Website](https://mahis.life/clip-fields/)]

 * **VIMA**:"VIMA: General Robot Manipulation with Multimodal Prompts", "arXiv, Oct 2022", [[Paper](https://arxiv.org/abs/2210.03094)] [[Pytorch Code](https://github.com/vimalabs/VIMA)] [[Website](https://vimalabs.github.io/)]

 * **Perceiver-Actor**:"A Multi-Task Transformer for Robotic Manipulation", *CoRL, Sep 2022*. [[Paper](https://peract.github.io/paper/peract_corl2022.pdf)] [[Pytorch Code](https://github.com/peract/peract)] [[Website](https://peract.github.io/)]

 * **LaTTe**: "LaTTe: Language Trajectory TransformEr", *arXiv, Aug 2022*. [[Paper](https://arxiv.org/abs/2208.02918)] [[TensorFlow Code](https://github.com/arthurfenderbucker/NL_trajectory_reshaper)] [[Website](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/robot-language/)]

 * **Robots Enact Malignant Stereotypes**: "Robots Enact Malignant Stereotypes", *FAccT, Jun 2022*. [[Paper](https://arxiv.org/abs/2207.11569)] [[Pytorch Code](https://github.com/ahundt/RobotsEnactMalignantStereotypes)] [[Website](https://sites.google.com/view/robots-enact-stereotypes/home)] [[Washington Post](https://www.washingtonpost.com/technology/2022/07/16/racist-robots-ai/)] [[Wired](https://www.wired.com/story/how-to-stop-robots-becoming-racist/)] (code access on request)

 * **ATLA**: "Leveraging Language for Accelerated Learning of Tool Manipulation", *CoRL, Jun 2022*. [[Paper](https://arxiv.org/abs/2206.13074)]

 * **ZeST**: "Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?", *L4DC, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.11134)]

 * **LSE-NGU**: "Semantic Exploration from Language Abstractions and Pretrained Representations", *arXiv, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.05080)]

 * **Embodied-CLIP**: "Simple but Effective: CLIP Embeddings for Embodied AI ", *CVPR, Nov 2021*. [[Paper](https://arxiv.org/abs/2111.09888)] [[Pytorch Code](https://github.com/allenai/embodied-clip)]

 * **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL, Sept 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Pytorch Code](https://github.com/cliport/cliport)] [[Website](https://cliport.github.io/)]

---
## Instructions and Navigation

 * **ADAPT**: "ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts", *CVPR, May 2022*. [[Paper](https://arxiv.org/abs/2205.15509)]

 * "The Unsurprising Effectiveness of Pre-Trained Vision Models for Control", *ICML, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.03580)] [[Pytorch Code](https://github.com/sparisi/pvr_habitat)] [[Website](https://sites.google.com/view/pvr-control)]

 * **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.10421)]

 * **Recurrent VLN-BERT**: "A Recurrent Vision-and-Language BERT for Navigation", *CVPR, Jun 2021* [[Paper](https://arxiv.org/abs/2011.13922)] [[Pytorch Code](https://github.com/YicongHong/Recurrent-VLN-BERT)]

 * **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV, Apr 2020* [[Paper](https://arxiv.org/abs/2004.14973)] [[Pytorch Code](https://github.com/arjunmajum/vln-bert)]

* "Interactive Language: Talking to Robots in Real Time", *arXiv, Oct 2022* [[Paper](https://arxiv.org/abs/2210.06407)] [[Website](https://interactive-language.github.io/)]

---
## Simulation Frameworks

 * **MineDojo**: "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge", *arXiv, Jun 2022*. [[Paper](https://arxiv.org/abs/2206.08853)] [[Code](https://github.com/MineDojo/MineDojo)] [[Website](https://minedojo.org/)] [[Open Database](https://minedojo.org/knowledge_base.html)]
 * **Habitat 2.0**: "Habitat 2.0: Training Home Assistants to Rearrange their Habitat", *NeurIPS, Dec 2021*. [[Paper](https://arxiv.org/abs/2106.14405)] [[Code](https://github.com/facebookresearch/habitat-sim)] [[Website](https://aihabitat.org/)]
 * **BEHAVIOR**: "BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments", *CoRL, Nov 2021*. [[Paper](https://arxiv.org/abs/2108.03332)] [[Code](https://github.com/StanfordVL/behavior)] [[Website](https://behavior.stanford.edu/)]
 * **iGibson 1.0**: "iGibson 1.0: a Simulation Environment for Interactive Tasks in Large Realistic Scenes", *IROS, Sep 2021*. [[Paper](https://arxiv.org/abs/2012.02924)] [[Code](https://github.com/StanfordVL/iGibson)] [[Website](https://svl.stanford.edu/igibson/)]
 * **ALFRED**: "ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks", *CVPR, Jun 2020*. [[Paper](https://arxiv.org/abs/1912.01734)] [[Code](https://github.com/askforalfred/alfred)] [[Website](https://askforalfred.com/)]
  * **BabyAI**: "BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning", *ICLR, May 2019*. [[Paper](https://openreview.net/pdf?id=rJeXCo0cYX)] [[Code](https://github.com/mila-iqia/babyai/tree/iclr19)]


----
## Citation
If you find this repository useful, please consider citing this list:
```
@misc{kira2022llmroboticspaperslist,
    title = {Awesome-LLM-Robotics},
    author = {Zsolt Kira},
    journal = {GitHub repository},
    url = {https://github.com/GT-RIPL/Awesome-LLM-Robotics},
    year = {2022},
}
```
