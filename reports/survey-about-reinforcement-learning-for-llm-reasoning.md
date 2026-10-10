# Reinforcement Learning for LLM Reasoning: A Survey

## Synthesis by theme

### 1. Scope and framing

Reinforcement learning (RL) for large language model (LLM) reasoning is best understood as post-training a model to produce reasoning trajectories that lead to externally assessable outcomes. In contrast to supervised fine-tuning (SFT), which imitates demonstrations, RL samples outputs from a policy and changes their likelihood according to rewards or preferences. The central design problem is therefore not simply choosing an optimizer: it is deciding what counts as a good reasoning trajectory, how to assign credit across a long sequence of tokens, and how to evaluate whether the model learned robust reasoning rather than a reward-specific shortcut.

The field spans a range of reasoning structures and search strategies—including chains, trees, graphs, nested reasoning, beam search, and Monte Carlo tree search—as well as policy/value learning and outcome- or process-based supervision. This breadth makes it useful to distinguish (i) reward and verifier design, (ii) policy optimization and rollout infrastructure, (iii) training regimes and inference-time compute, and (iv) evaluation and generalization. [1]

### 2. What is being rewarded: outcomes, processes, and verifiers

A foundational distinction is between **outcome supervision**, which scores the final answer, and **process supervision**, which assesses intermediate steps. In the MATH setting, process supervision outperformed outcome supervision in the reported comparison; the process-supervised system solved 78.2% of a representative test subset, and the work released PRM800K with 800,000 step-level human feedback labels. Step-level feedback can identify where a solution first goes wrong, while an answer checker only establishes whether the final answer is correct. But process labels are expensive, step boundaries can be ambiguous, and there is no generally simple automatic equivalent to human judgment of each reasoning step. [2]

A prominent alternative is **reinforcement learning with verifiable rewards (RLVR)**. For domains such as exact-answer mathematics and executable code, a rule-based verifier can return a correctness signal without training a general-purpose reward model. DeepSeekMath introduced GRPO in a mathematical-reasoning training setup, and DeepSeek-R1-Zero later demonstrated large-scale RL directly from a base model using rule-based answer correctness and formatting rewards. DeepSeek describes using answer-format checks for deterministic math and compiler/test-case checks for code. R1-Zero intentionally did not require a prescribed reasoning process, allowing the policy to discover its own trajectories. [3][4]

Rule-based verification is attractive because it can be objective and scalable where answers are machine-checkable, but a correct final answer does not establish that the reasoning is valid. Model-based rewards can judge less formal outputs and support general alignment, but introduce their own modeling, compute, and reward-hacking risks. In the R1 report, the authors avoided neural outcome- and process-reward models for reasoning RL, citing reward hacking concerns, retraining expense, and pipeline complexity. They also discuss limitations of process reward models: difficulty defining and labeling steps, unsatisfactory automated annotation, and limited benefit relative to added cost in their large-scale experiments. [4]

### 3. Policy optimization and training-system choices

The PPO/GRPO contrast illustrates a key systems trade-off. PPO typically learns a value function and uses Generalized Advantage Estimation; this adds model-sized memory and compute, and assigning token-level values is difficult when feedback arrives only at the end of a long response. GRPO removes the separate critic and estimates a baseline from the relative scores of multiple responses sampled for the same prompt. It retains a clipped policy objective and a reference-policy KL term. DeepSeekMath reports that its GRPO phase improved DeepSeekMath-Instruct from 82.9% to 88.2% on GSM8K, 46.8% to 51.7% on MATH, and 84.6% to 88.8% on CMATH. [3]

GRPO is not a complete solution to training stability or efficiency. Long reasoning traces make online rollouts costly, and groups in which every sample is right or every sample is wrong provide little relative learning signal. DAPO addresses system-level issues with four named mechanisms: Clip-Higher to encourage diversity and avoid entropy collapse, Dynamic Sampling to retain prompts with mixed sampled outcomes, token-level policy-gradient loss for long CoTs, and Overlong Reward Shaping to handle truncated generations. In the paper's Qwen2.5-32B setup, the reported AIME score rose from 30 for a naive GRPO baseline to 50 with the full DAPO system. This is a result for that particular model, training setup, and benchmark—not a general algorithm ranking. [5]

Open-R1's reproduction work underscores the operational costs and reproducibility questions behind such results. It reports that long completions make GRPO memory-intensive and that generation is a major online-training bottleneck; its implementation work integrated distributed training and vLLM generation. The team reproduced several DeepSeek distilled-model MATH-500 figures within roughly 1–3 standard deviations, while noting that pass@1 depends on sampling counts and evaluation protocol. [6]

### 4. Training regimes: from pure RL to staged post-training

The empirical trajectory of the field includes both RL-first and staged approaches. DeepSeek-R1-Zero was trained from DeepSeek-V3-Base without preliminary SFT. The paper reports AIME 2024 pass@1 increasing from 15.6% initially to 71.0% after RL steps, and 86.7% with 64-sample majority voting. It describes self-verification, reflection, and longer reasoning traces emerging during training, alongside undesirable readability, repetition, and language-mixing behavior. [4]

DeepSeek-R1 adds a more structured recipe: cold-start examples, reasoning-focused RL, rejection sampling to create further reasoning data, SFT over reasoning and general-purpose data, and another RL phase for broader alignment and reasoning. The HF Papers entry emphasizes the distinction between R1-Zero's RL without preliminary SFT and R1's cold-start/multi-stage pipeline, and records the release of six dense distilled models based on Qwen and Llama spanning 1.5B to 70B. [4][7]

Other systems emphasize complementary choices. OpenAI describes o1 as trained with large-scale RL and reports improvements with both more RL training compute and more test-time thinking compute. Its post reports 74% AIME 2024 pass@1, 83% with consensus@64, and 93% when reranking 1,000 samples with a learned scorer; those figures illustrate how strongly reported performance can depend on inference-time sampling and selection. [8]

Kimi k1.5 studies long-context RL, reporting context scaling to 128K, partial rollouts, asynchronous rollout workers, and a policy-optimization approach without MCTS, a value function, or a process reward model. Its paper associates longer context with more search steps and reports learned behaviors such as planning, reflection, and correction. The HF Daily Papers summary similarly highlights long-context scaling and improved policy optimization as central choices. [9][10]

QwQ-32B represents another open-weight recipe: it begins from a cold-start checkpoint, uses outcome-based rewards and math/code verifiers in an initial RL stage, then adds a short general-RL stage using a general reward model and rule-based verification. Qwen reports the latter improved instruction following, preference alignment, and agent performance without significant math/coding degradation. The HF model card lists 32.5B parameters and a 131,072-token context, illustrating how long-context reasoning models are also packaged and documented for downstream use. [11][12]

### 5. Scaling, distillation, and inference-time reasoning

A recurring empirical finding is that reasoning performance can be scaled along more than one axis. OpenAI's o1 report explicitly highlights both training-time RL compute and test-time thinking compute. Kimi k1.5 reports that response length and accuracy rose during RL training, and that its long-CoT model scored 77.5 on AIME and 96.2 on MATH-500; its long2short RL phase reported AIME 60.8 with an average of 3,272 generated tokens. These results make reasoning-token budget and context length important system variables, not merely incidental output properties. [8][9]

Distillation offers a route to transfer reasoning behavior into smaller models. DeepSeek reports that direct distillation from R1 outperformed more than 10,000 RL steps on Qwen-32B-Base in its comparison. The distilled Qwen-32B scored 72.6% AIME pass@1 and 94.3% MATH-500, compared with 47.0% and 91.6% for R1-Zero-Qwen-32B in the reported table. The HF model repository says the Qwen distilled models were fine-tuned using 800K samples curated with R1. [4][7]

Inference-time scaling is related to, but distinct from, RL training. The s1 approach combines SFT on a 1,000-question reasoning dataset with “budget forcing,” which can extend thinking by suppressing an end-of-thinking marker or shorten it by forcing termination. Its HF Papers page reports improvements on competition math and notes that gains eventually flatten with repeated extension, while excessive suppression can cause loops. This is a useful comparison: longer reasoning can improve performance, but controlling its duration is itself a policy and evaluation problem. [13]

### 6. Evaluation: what benchmark scores do and do not show

AIME, MATH, coding, and science benchmarks have become common measures, but scores are not directly comparable without matching model versions, decoding, sample counts, and scoring protocols. For example, DeepSeek documents a 32,768-token generation cap and its temperature, top-p, and sampling settings; Open-R1 reports different per-benchmark sample counts for its reproduction and notes that sampling counts affect pass@1 estimates. Benchmark tables should therefore be read together with evaluation conditions, not as protocol-independent rankings. [4][6]

Pass@1 and majority-vote performance also measure different capabilities. o1's reported AIME score rises from 74% pass@1 to 83% with consensus@64 and 93% with learned reranking over 1,000 samples. Such gains show the value of test-time compute and selection, but do not alone establish that a single sampled reasoning trace is more reliable or faithful. [8]

A broader empirical study of RLVR reports improved performance at small sample counts but lower large-k Pass@k than base models in its tested settings. It interprets the pattern as improved sampling efficiency toward successful paths already present in the base model's distribution, potentially accompanied by narrower reasoning coverage. The authors also caution that a correct final answer can pair with invalid intermediate reasoning, recommending large-k evaluation, coverage measures, and chain-of-thought validity checks as complementary lenses. These findings are limited to the study's tested model families, algorithms, and math, coding, and visual-reasoning tasks. [14]

### 7. Failure modes and open research challenges

**Reward hacking and verifier quality.** A verifier can be formally correct about the output it checks yet fail to measure the intended reasoning. Outcome-only rewards may incentivize answer-producing shortcuts; a learned reward model can itself be overoptimized. DeepSeek's concerns about PRM hacking and complexity reflect a general tension: richer supervision can provide more signal but also introduces more places where a proxy can be exploited. [2][4]

**Credit assignment and reasoning faithfulness.** Sparse terminal rewards make it hard to identify which decisions in a long chain caused success. Process supervision can expose local errors, but costs human effort and requires decisions about step boundaries and labels. Conversely, a correct final answer may have come from a flawed chain, and outcome reward cannot by itself certify that a trajectory represents a sound proof or explanation. [2][4][14]

**Overthinking, length control, and efficiency.** Long contexts and extended rollouts can facilitate search and reflection, but consume generation and training resources. Kimi's long2short phase and s1's budget forcing both treat reasoning length as something to optimize rather than maximize blindly. DAPO's overlong shaping and Open-R1's discussion of memory and generation bottlenecks show that the practical problem includes throughput, memory, truncation, and token efficiency as well as benchmark accuracy. [9][5][6][13]

**Generalization versus concentration of probability mass.** RL can make correct responses more likely in sampled outputs without necessarily expanding the range of valid strategies the model can produce. The reported large-k results and chain-validity concerns motivate testing generalization to shifted tasks and checking solution structure, not just final-answer accuracy. Distilled and benchmark-specialized systems should likewise be evaluated across formats and domains rather than inferred to have general reasoning from a narrow score suite. [14][4]

**Reproducibility and infrastructure.** Training details that materially affect performance include prompt and data selection, rollout length, sampling, reward normalization, tokenization, optimization, and evaluation. Open-R1's reproduction work shows that even reproducing a reported benchmark requires matching evaluation protocols, while long-context training and inference impose substantial engineering costs. The DAPO and Open-R1 reports make release of code/data and transparent protocol descriptions important complements to model claims. [5][6][15]

### 8. Synthesis and outlook

RL has become a powerful method for improving LLM reasoning when tasks supply rewards that can be checked at scale. The strongest practical recipe is not singular: evidence in this survey ranges from SFT-free RL exploration to cold-start and multi-stage training, critic-free group-relative optimization, verifier-based outcome rewards, process supervision, and distillation. The research frontier is consequently about matching the reward signal and optimization system to the task while preserving generalization, interpretability of evaluation, and affordable inference.

Three implications follow. First, verifiability is a major enabler, but answer correctness is an incomplete proxy for reasoning quality. Second, training compute, test-time compute, and reasoning-token budgets interact; score improvements should be reported with their sampling and token costs. Third, credible progress requires evaluation beyond a single pass@1 number: robust protocols should include diverse sampling regimes, reasoning validity or coverage diagnostics where feasible, and careful reporting of verifiers, data, and infrastructure. These are not reasons to discount RL's demonstrated gains, but a research agenda for distinguishing genuine, transferable reasoning from optimization to a narrow reward.

## References
[1] Reasoning Language Models: A Blueprint. hf-daily. https://huggingface.co/papers/2501.11223 (2025-01-20)
[2] Let's Verify Step by Step. arXiv. https://arxiv.org/abs/2305.20050 (2023-05-31)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv. https://arxiv.org/abs/2402.03300 (2024-02-05)
[4] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[5] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv. https://arxiv.org/html/2503.14476v1 (2025-03-18)
[6] Open-R1: Update #1. web. https://huggingface.co/blog/open-r1/update-1 (2025-02-02)
[7] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-daily. https://huggingface.co/papers/2501.12948 (2025-01-23)
[8] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[9] Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv. https://arxiv.org/abs/2501.12599 (2025-01-28)
[10] Kimi k1.5: Scaling Reinforcement Learning with LLMs. hf-daily. https://huggingface.co/papers/2501.12599 (2025-01-23)
[11] QwQ-32B: Embracing the Power of Reinforcement Learning. web. https://qwenlm.github.io/blog/qwq-32b/ (2025-03-06)
[12] Qwen/QwQ-32B model card. web. https://huggingface.co/Qwen/QwQ-32B (2025-03)
[13] s1: Simple test-time scaling. hf-search. https://huggingface.co/papers/2501.19393 (2025-01-31)
[14] The effect of current RLVR on LLM's reasoning ability. web. https://arxiv.org/html/2504.13837v3 (2025)
[15] Open R1: Update #3. web. https://huggingface.co/blog/open-r1/update-3 (2025-03-11)
