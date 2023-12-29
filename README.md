# Awesome-LLM-Robotics [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo contains a curative list of **papers using Large Language/Multi-Modal Models for Robotics/RL**. Template from [awesome-Implicit-NeRF-Robotics](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics) <br>

#### Please feel free to send me [pull requests](https://github.com/GT-RIPL/Awesome-LLM-Robotics/blob/main/how-to-PR.md) or [email](mailto:zkira-changetoat-gatech--changetodot-changetoedu) to add papers! <br>

If you find this repository useful, please consider [citing](#citation) and STARing this list. Feel free to share this list with others!

---
## Overview

  - [Surveys](#surveys)
  - [Reasoning](#reasoning)
  - [Planning](#planning)
  - [Manipulation](#manipulation)
  - [Instructions and Navigation](#instructions-and-navigation)
  - [Simulation Frameworks](#simulation-frameworks)
  - [Citation](#citation)

---
## Surveys
* "Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis", *arXiv, Dec 2023*. [[Paper](https://arxiv.org/abs/2312.08782)] [[Paper List](https://github.com/JeffreyYH/robotics-fm-survey)] [[Website](https://robotics-fm-survey.github.io/)] 
* "Language-conditioned Learning for Robotic Manipulation: A Survey", *arXiv, Dec 2023*, [[Paper](https://arxiv.org/abs/2312.10807)] 
* "Foundation Models in Robotics: Applications, Challenges, and the Future", *arXiv, Dec 2023*, [[Paper](https://arxiv.org/abs/2312.07843)] [[Paper List](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models)]
* "Robot Learning in the Era of Foundation Models: A Survey", *arXiv, Nov 2023*, [[Paper](https://arxiv.org/abs/2311.14379)]
* "The Development of LLMs for Embodied Navigation", *arXiv, Nov 2023*, [[Paper](https://arxiv.org/abs/2311.00530)]
---
## Reasoning

* **LEO**: "An Embodied Generalist Agent in 3D World", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.12871)] [[Code](https://github.com/embodied-generalist/embodied-generalist)] [[Website](https://embodied-generalist.github.io/)]
* **Robogen**: "A generative and self-guided robotic agent that endlessly propose and master new skills.", *arXiv, Nov 2023*. [[Paper](https://arxiv.org/abs/2311.01455)] [[Code](https://github.com/Genesis-Embodied-AI/RoboGen)] [[Website](https://robogen-ai.github.io/)] 
* **LLaRP**: "Large Language Models as Generalizable Policies for Embodied Tasks", *arXiv, Oct 2023*. [[Paper](https://arxiv.org/abs/2310.17722)] [[Website](https://llm-rl.github.io)]
* **RT-X**: "Open X-Embodiment: Robotic Learning Datasets and RT-X Models", *arXiv, July 2023*. [[Paper](https://robotics-transformer-x.github.io/paper.pdf)] [[Website](https://robotics-transformer-x.github.io/)]
* **RT-2**: "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", *arXiv, July 2023*. [[Paper](https://arxiv.org/abs/2307.15818)] [[Website](https://robotics-transformer2.github.io/)]
* **Instruct2Act**: "Mapping Multi-modality Instructions to Robotic Actions with Large Language Model", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.11176)]  [[Pytorch Code](https://github.com/OpenGVLab/Instruct2Act)]
* **TidyBot**: "Personalized Robot Assistance with Large Language Models",  *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.05658)] [[Pytorch Code](https://github.com/jimmyyhwu/tidybot/tree/main/robot)] [[Website](https://tidybot.cs.princeton.edu/)]
* **Generative Agents**: "Generative Agents: Interactive Simulacra of Human Behavior", *arXiv, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.03442v1) [Code](https://github.com/joonspk-research/generative_agents)] 
* **CortexBench**: "Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?" *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2303.18240)]
* **Matcha**: "Chat with the Environment: Interactive Multimodal Perception using Large Language Models", *IROS, Mar 2023*. [[Paper](https://arxiv.org/abs/2303.08268)] [[Github](https://github.com/xf-zhao/Matcha)] [[Website](https://matcha-model.github.io/)]
* **PaLM-E**: "PaLM-E: An Embodied Multimodal Language Model", *arXiv, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.03378)] [[Webpage](https://palm-e.github.io/)]
* "Large Language Models as Zero-Shot Human Models for Human-Robot Interaction", *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2303.03548v1)] 
* "Translating Natural Language to Planning Goals with Large-Language Models", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.05128)] 
* **RT-1**: "RT-1: Robotics Transformer for Real-World Control at Scale", *arXiv, Dec 2022*. [[Paper](https://arxiv.org/abs/2212.06817)]  [[GitHub](https://github.com/google-research/robotics_transformer)] [[Website](https://robotics-transformer.github.io/)]
* "PDDL Planning with Pretrained Large Language Models", *NeurlPS, Oct 2022*. [[Paper](https://openreview.net/forum?id=1QMMUB4zfl)] [[Github](https://tinyurl.com/llm4pddl)]
* **ProgPrompt**: "Generating Situated Robot Task Plans using Large Language Models", *arXiv, Sept 2022*. [[Paper](https://arxiv.org/abs/2209.11302)]  [[Github](https://github.com/progprompt/progprompt)] [[Website](https://progprompt.github.io/)]
* **Code-As-Policies**: "Code as Policies: Language Model Programs for Embodied Control", *arXiv, Sept 2022*. [[Paper](https://arxiv.org/abs/2209.07753)]  [[Colab](https://github.com/google-research/google-research/tree/master/code_as_policies)] [[Website](https://code-as-policies.github.io/)]
* **PIGLeT**: "PIGLeT: Language Grounding Through Neuro-Symbolic Interaction in a 3D World", *ACL, June 2021*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](http://github.com/rowanz/piglet)] [[Website](https://rowanzellers.com/piglet/)]
* **Say-Can**: "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.01691)]  [[Colab](https://say-can.github.io/#open-source)] [[Website](https://say-can.github.io/)]
* **Socratic**: "Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language", *arXiv, Apr 2021*. [[Paper](https://arxiv.org/abs/2204.00598)] [[Pytorch Code](https://socraticmodels.github.io/#code)] [[Website](https://socraticmodels.github.io/)]


---
## Planning
* **ViLa**: "Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning", *arXiv, Sept 2023*, [[Paper](https://arxiv.org/abs/2311.17842)] [[Website](https://robot-vila.github.io/)]
* **LGMCTS**: "LGMCTS: Language-Guided Monte-Carlo Tree Search for Executable Semantic Object Rearrangement", *arXiv, Sept 2023*. [[Paper](https://arxiv.org/abs/2309.15821)]
* **Prompt2Walk**: "Prompt a Robot to Walk with Large Language Models", *arXiv, Sept 2023*, [[Paper](https://arxiv.org/abs/2309.09969)] [[Website](https://prompt2walk.github.io)]
* **DoReMi**: "Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", *arXiv, July 2023*, [[Paper](https://arxiv.org/abs/2307.00329)] [[Website](https://sites.google.com/view/doremi-paper)]
* **Co-LLM-Agents**: "Building Cooperative Embodied Agents Modularly with Large Language Models", *arXiv, July 2023*. [[Paper](https://arxiv.org/abs/2307.02485)] [[Code](https://github.com/UMass-Foundation-Model/Co-LLM-Agents)] [[Website](https://vis-www.cs.umass.edu/Co-LLM-Agents/)]
* **LLM-Reward**: "Language to Rewards for Robotic Skill Synthesis", *arXiv, June 2023*. [[Paper](https://arxiv.org/abs/2306.08647)] [[Website](https://language-to-reward.github.io/)]
* **GLAM**: "Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2302.02662)] [[Pytorch Code](https://github.com/flowersteam/Grounding_LLMs_with_online_RL)] 
* **LLM-BRAIn**: "LLM-BRAIn: AI-driven Fast Generation of Robot Behaviour Tree based on Large Language Model", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.19352)]
* **LLM-MCTS**: "Large Language Models as Commonsense Knowledge for Large-Scale Task Planning", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.14078v1)] 
* **LLM+P**: "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency", *arXiv, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.11477)] [[Code](https://github.com/Cranial-XIX/llm-pddl)]
* **ChatGPT-Prompts**: "ChatGPT Empowered Long-Step Robot Control in Various Environments: A Case Application", *arXiv, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.03893?s=03)] [[Code/Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts)]
* **LLM-Brain**: "LLM as A Robotic Brain: Unifying Egocentric Memory and Control", arXiv, Apr 2023. [[Paper](https://arxiv.org/abs/2304.09349v1)] 
* **LLM-planner**: "LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models", *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2212.04088)] [[Pytorch Code](https://github.com/OSU-NLP-Group/LLM-Planner/)] [[Website](https://dki-lab.github.io/LLM-Planner/)]
* "Foundation Models for Decision Making: Problems, Methods, and Opportunities", *arXiv, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.04129)]
* **Text2Motion**: "Text2Motion: From Natural Language Instructions to Feasible Plans", *arXiV, Mar 2023*, [[Paper](https://arxiv.org/abs/2303.12153)] [[Website](https://sites.google.com/stanford.edu/text2motion)]
* **GD**: "Grounded Decoding: Guiding Text Generation with Grounded Models for Robot Control", *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2303.00855)] [[Website](https://grounded-decoding.github.io/)]
* "Reward Design with Language Models", *ICML, Feb 2023*. [[Paper](https://arxiv.org/abs/2303.00001v1)] [[Pytorch Code](https://github.com/minaek/reward_design_with_llms)] 
* **PromptCraft**: "ChatGPT for Robotics: Design Principles and Model Abilities", *Blog, Feb 2023*, [[Paper](https://arxiv.org/abs/2306.17582)] [[Website](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/chatgpt-for-robotics/)]
* "Planning with Large Language Models via Corrective Re-prompting", *arXiv, Nov 2022*. [[Paper](https://arxiv.org/abs/2311.09935)]
* **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", *ICLR, 2023*. [[Paper](https://arxiv.org/abs/2210.03629)] [[Github](https://github.com/ysymyth/ReAct)] [[Website](https://react-lm.github.io/)] <!-- Oct 2022 on arXiv -->
* **Don't Copy the Teacher**: "Don’t Copy the Teacher: Data and Model Challenges in Embodied Dialogue", *EMNLP, 2022*. [[Paper](https://arxiv.org/abs/2210.04443)] [[Website](https://www.youtube.com/watch?v=qGPC65BDJw4&t=2s)] <!-- Oct 2022 on arXiv -->
* **COWP**: "Robot Task Planning and Situation Handling in Open Worlds", *arXiv, Oct 2022*. [[Paper](https://arxiv.org/abs/2210.01287)] [[Pytorch Code](https://github.com/yding25/GPT-Planner)] [[Website](https://cowplanning.github.io/)]
* **LM-Nav**: "Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.04429)] [[Pytorch Code](https://github.com/blazejosinski/lm_nav)] [[Website](https://sites.google.com/view/lmnav)]
* **InnerMonlogue**: "Inner Monologue: Embodied Reasoning through Planning with Language Models", *arXiv, July 2022*. [[Paper](https://arxiv.org/abs/2207.05608)] [[Website](https://innermonologue.github.io/)]
* **Housekeep**: "Housekeep: Tidying Virtual Households using Commonsense Reasoning", *arXiv, May 2022*. [[Paper](https://arxiv.org/abs/2205.10712)] [[Pytorch Code](https://github.com/yashkant/housekeep)] [[Website](https://yashkant.github.io/housekeep/)]
* **MOO**: "Open-World Object Manipulation using Pre-Trained Vision-Language Models", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2303.00905)] [[Website](https://robot-moo.github.io/)]
* **LID**: "Pre-Trained Language Models for Interactive Decision-Making", *arXiv, Feb 2022*. [[Paper](https://arxiv.org/abs/2202.01771)] [[Pytorch Code](https://github.com/ShuangLI59/Language-Model-Pre-training-Improves-Generalization-in-Policy-Learning)] [[Website](https://shuangli-project.github.io/Pre-Trained-Language-Models-for-Interactive-Decision-Making/)]
* "Collaborating with language models for embodied reasoning", *NeurIPS, Feb 2022*. [[Paper](https://arxiv.org/abs/2302.00763v1)]
* **ZSP**: "Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents", *ICML, Jan 2022*. [[Paper](https://arxiv.org/abs/2201.07207)] [[Pytorch Code](https://github.com/huangwl18/language-planner)] [[Website](https://wenlong.page/language-planner/)]
* **FILM**: "FILM: Following Instructions in Language with Modular Methods", *ICLR, 2022*. [[Paper](https://arxiv.org/abs/2110.07342)] [[Code](https://github.com/soyeonm/FILM)] [[Website](https://soyeonm.github.io/FILM_webpage/)] <!-- Oct 2021 on arXiv -->
* **CALM**: "Keep CALM and Explore: Language Models for Action Generation in Text-based Games", *arXiv, Oct 2020*. [[Paper](https://arxiv.org/abs/2010.02903)] [[Pytorch Code](https://github.com/princeton-nlp/calm-textgame)] 
* "Visually-Grounded Planning without Vision: Language Models Infer Detailed Plans from High-level Instructions", *arXiV, Oct 2020*, [[Paper](https://arxiv.org/abs/2009.14259)] 


---
## Manipulation

* **BOSS**: "Bootstrap Your Own Skills: Learning to Solve New Tasks with LLM Guidance", *CoRL, Nov 2023*. [[Paper](https://openreview.net/forum?id=a0mFRgadGO)] [[Website](https://clvrai.github.io/boss/)]
* **Octopus**: "Octopus: Embodied Vision-Language Programmer from Environmental Feedback", *arXiv, Oct 2023*, [[Paper](https://arxiv.org/abs/2310.08588)] [[PyTorch Code](https://github.com/dongyh20/Octopus)] [[Website](https://choiszt.github.io/Octopus/)]
* **PhysObjects**: "Physically Grounded Vision-Language Models for Robotic Manipulation", *arxiv, Sept 2023*. [[Paper](https://arxiv.org/abs/2309.02561)]
* **Text2Reward**: "Text2Reward: Automated Dense Reward Function Generation for Reinforcement Learning", *arXiv, Sept 2023*, [[Paper](https://arxiv.org/abs/2309.11489)] [[Website](https://text-to-reward.github.io/)]
* **Scalingup**: "Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition", *arXiv, July 2023*. [[Paper](https://arxiv.org/abs/2307.14535)] [[Code](https://github.com/columbia-ai-robotics/scalingup)] [[Website](https://www.cs.columbia.edu/~huy/scalingup/)]
* **VoxPoser**:"VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models", *arXiv, July 2023*. [[Paper](https://arxiv.org/abs/2307.05973)] [[Website](https://voxposer.github.io/)]
* **RoboCat**: "RoboCat: A self-improving robotic agent", *arxiv, June 2023*. [[Paper/PDF](https://arxiv.org/abs/2306.11706)]  [[Website](https://www.deepmind.com/blog/robocat-a-self-improving-robotic-agent)]
* **SPRINT**: "SPRINT: Semantic Policy Pre-training via Language Instruction Relabeling", *arXiv, June 2023*. [[Paper](https://arxiv.org/abs/2306.11886)] [[Website](https://clvrai.github.io/sprint/)]
* "Language Instructed Reinforcement Learning for Human-AI Coordination", *arXiv, June 2023*. [[Paper](https://arxiv.org/pdf/2304.07297)] 
* **LIV**: "LIV: Language-Image Representations and Rewards for Robotic Control", *arXiv, June 2023*, [[Paper](https://arxiv.org/abs/2306.00958)] [[Pytorch Code](https://github.com/penn-pal-lab/LIV)] [[Website](https://penn-pal-lab.github.io/LIV/)]
* **VOYAGER**: "VOYAGER: An Open-Ended Embodied Agent with Large Language Models", *arXiv, May 2023*. [[Paper](https://arxiv.org/abs/2305.16291)] [[Pytorch Code](https://github.com/MineDojo/Voyager)] [[Website](https://voyager.minedojo.org/)]
* **LLM-GROP**: "Task and Motion Planning with Large Language Models for Object Rearrangement", *arXiv, May 2023*. [[Paper](https://arxiv.org/pdf/2303.06247)] [[Website](https://sites.google.com/view/llm-grop)]
* **ProgramPort**: "Programmatically Grounded, Compositionally Generalizable Robotic Manipulation", *ICLR, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.13826)] [[Website] (https://progport.github.io/)]
* **CoTPC**: "Chain-of-Thought Predictive Control", *arXiv, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.00776)] [[Code](https://github.com/SeanJia/CoTPC)]
* **VLaMP**: "Pretrained Language Models as Visual Planners for Human Assistance", *arXiV, Apr 2023*, [[Paper](https://arxiv.org/abs/2304.09179)]
* "Towards a Unified Agent with Foundation Models", *ICLR, Mar 2023*. [[Paper](https://www.semanticscholar.org/paper/TOWARDS-A-UNIFIED-AGENT-WITH-FOUNDATION-MODELS-Palo-Byravan/67188a50e1d8a601896f1217451b99f646af4ac8)] 
* **Plan4MC**: "Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks", *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2303.16563)] [[Pytorch Code](https://github.com/PKU-RL/Plan4MC)] [[Website](https://sites.google.com/view/plan4mc)]
* **DEPS**: "Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.01560)] [[Pytorch Code](https://github.com/CraftJarvis/MC-Planner)]
* **ELLM**: "Guiding Pretraining in Reinforcement Learning with Large Language Models", *arXiv, Feb 2023*. [[Paper](https://arxiv.org/abs/2302.06692)] 
* **LILAC**: "No, to the Right – Online Language Corrections for Robotic Manipulation via Shared Autonomy", *arXiv, Jan 2023*, [[Paper](https://arxiv.org/abs/2301.02555)] [[Pytorch Code](https://github.com/Stanford-ILIAD/lilac)]
* **Gato**: "A Generalist Agent", *TMLR, Nov 2022*. [[Paper/PDF](https://openreview.net/pdf?id=1ikK0kHjvj)]  [[Website](https://www.deepmind.com/publications/a-generalist-agent)]
* **R3M**: "R3M: A Universal Visual Representation for Robot Manipulation", *arXiv, Nov 2022*, [[Paper](https://arxiv.org/abs/2203.12601)] [[Pytorch Code](https://github.com/facebookresearch/r3m)] [[Website](https://tinyurl.com/robotr3m)]
* **DIAL**: "Robotic Skill Acquisition via Instruction Augmentation with Vision-Language Models", *arXiv, Nov 2022*, [[Paper](https://arxiv.org/abs/2211.11736)] [[Website](https://instructionaugmentation.github.io/)]
* **CLIP-Fields**: "CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory", *arXiv, Oct 2022*, [[Paper](https://arxiv.org/abs/2210.05663)] [[PyTorch Code](https://github.com/notmahi/clip-fields)] [[Website](https://mahis.life/clip-fields/)]
* **VIMA**: "VIMA: General Robot Manipulation with Multimodal Prompts", *arXiv, Oct 2022*, [[Paper](https://arxiv.org/abs/2210.03094)] [[Pytorch Code](https://github.com/vimalabs/VIMA)] [[Website](https://vimalabs.github.io/)]
* **Perceiver-Actor**: "A Multi-Task Transformer for Robotic Manipulation", *CoRL, Sept 2022*. [[Paper](https://arxiv.org/abs/2209.05451)] [[Pytorch Code](https://github.com/peract/peract)] [[Website](https://peract.github.io/)]
* **NLMap**: "Open-vocabulary Queryable Scene Representations for Real World Planning", *arXiv, Sept 2022*, [[Paper](https://arxiv.org/abs/2209.09874)] [[Website](https://nlmap-saycan.github.io/)]
* **LaTTe**: "LaTTe: Language Trajectory TransformEr", *arXiv, Aug 2022*. [[Paper](https://arxiv.org/abs/2208.02918)] [[TensorFlow Code](https://github.com/arthurfenderbucker/NL_trajectory_reshaper)] [[Website](https://www.microsoft.com/en-us/research/group/autonomous-systems-group-robotics/articles/robot-language/)]
* **Robots Enact Malignant Stereotypes**: "Robots Enact Malignant Stereotypes", *FAccT, June 2022*. [[Paper](https://arxiv.org/abs/2207.11569)] [[Pytorch Code](https://github.com/ahundt/RobotsEnactMalignantStereotypes)] [[Website](https://sites.google.com/view/robots-enact-stereotypes/home)] [[Washington Post](https://www.washingtonpost.com/technology/2022/07/16/racist-robots-ai/)] [[Wired](https://www.wired.com/story/how-to-stop-robots-becoming-racist/)] (code access on request)
* **ATLA**: "Leveraging Language for Accelerated Learning of Tool Manipulation", *CoRL, June 2022*. [[Paper](https://arxiv.org/abs/2206.13074)]
* **ZeST**: "Can Foundation Models Perform Zero-Shot Task Specification For Robot Manipulation?", *L4DC, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.11134)]
* **LSE-NGU**: "Semantic Exploration from Language Abstractions and Pretrained Representations", *arXiv, Apr 2022*. [[Paper](https://arxiv.org/abs/2204.05080)]
* **MetaMorph**: "METAMORPH: LEARNING UNIVERSAL CONTROLLERS WITH TRANSFORMERS", *arxiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.11931)]
* **Embodied-CLIP**: "Simple but Effective: CLIP Embeddings for Embodied AI", *CVPR, Nov 2021*. [[Paper](https://arxiv.org/abs/2111.09888)] [[Pytorch Code](https://github.com/allenai/embodied-clip)]
* **CLIPort**: "CLIPort: What and Where Pathways for Robotic Manipulation", *CoRL, Sept 2021*. [[Paper](https://arxiv.org/abs/2109.12098)] [[Pytorch Code](https://github.com/cliport/cliport)] [[Website](https://cliport.github.io/)]
* **TIP**: "Multimodal Procedural Planning via Dual Text-Image Prompting", *arXiV, May 2023*, [[Paper](https://arxiv.org/abs/2305.01795)]


---
## Instructions and Navigation

* **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", *CoRL, Nov 2023*. [[Paper](https://openreview.net/forum?id=cjEI5qXoT0)] [[Code](https://github.com/changhaonan/OVSG)] [[Website](https://ovsg-l.github.io/)]
* **VLMaps**: "Visual Language Maps for Robot Navigation", *arXiv, Mar 2023*. [[Paper](https://arxiv.org/abs/2210.05714)] [[Pytorch Code](https://github.com/vlmaps/vlmaps)] [[Website](https://vlmaps.github.io/)]
* "Interactive Language: Talking to Robots in Real Time", *arXiv, Oct 2022* [[Paper](https://arxiv.org/abs/2210.06407)] [[Website](https://interactive-language.github.io/)]
 * **NLMap**: "Open-vocabulary Queryable Scene Representations for Real World Planning", *arXiv, Sep 2022*, [[Paper](https://arxiv.org/abs/2209.09874)] [[Website](https://nlmap-saycan.github.io/)]
* **ADAPT**: "ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts", *CVPR, May 2022*. [[Paper](https://arxiv.org/abs/2205.15509)]
* "The Unsurprising Effectiveness of Pre-Trained Vision Models for Control", *ICML, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.03580)] [[Pytorch Code](https://github.com/sparisi/pvr_habitat)] [[Website](https://sites.google.com/view/pvr-control)]
* **CoW**: "CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration", *arXiv, Mar 2022*. [[Paper](https://arxiv.org/abs/2203.10421)]
* **Recurrent VLN-BERT**: "A Recurrent Vision-and-Language BERT for Navigation", *CVPR, June 2021* [[Paper](https://arxiv.org/abs/2011.13922)] [[Pytorch Code](https://github.com/YicongHong/Recurrent-VLN-BERT)]
* **VLN-BERT**: "Improving Vision-and-Language Navigation with Image-Text Pairs from the Web", *ECCV, Apr 2020* [[Paper](https://arxiv.org/abs/2004.14973)] [[Pytorch Code](https://github.com/arjunmajum/vln-bert)]


---
## Simulation Frameworks
 * **GENESIS**: "A generative world for general-purpose robotics & embodied AI learning.", *arXiv, Nov 2023*. [[Code](https://github.com/Genesis-Embodied-AI/Genesis)] 
 * **ARNOLD**: "ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes", *ICCV, Apr 2023*. [[Paper](https://arxiv.org/abs/2304.04321)] [[Code](https://github.com/arnold-benchmark/arnold)] [[Website](https://arnold-benchmark.github.io/)]
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
