
# Research Protocol: Evaluating False Positive Rates in Multi-Agent AI Juries vs. Human Juries

This document outlines the theoretical background, architectural framework, and experimental methodology for a study comparing the rate of false-positive verdicts between multi-agent Large Language Model (LLM) juries and real-world human juries using local open-source systems.

## 1. Research Objectives & Hypotheses

### Primary Research Question

_Do multi-agent AI juries, when operating under explicit instructions prioritizing the presumption of innocence (Blackstone's Ratio), yield a higher or lower rate of false-positive convictions compared to human juries exposed to the identical trial record?_

### Key Hypotheses

-   **Hypothesis 1 (**$H_1$**):** Homogeneous local AI juries (using identical baseline system prompts) will exhibit severe groupthink and a higher false-positive rate compared to heterogeneous local AI juries utilizing randomized cognitive personas.
    
-   **Hypothesis 2 (**$H_2$**):** Without explicit structural prompts enforcing _reasonable doubt_ thresholds at each phase of deliberation, local open-source models will display "anti-Bayesian drift" (overconfidence in their initial, zero-shot readings of the transcript, ignoring counter-arguments).
    
-   **Hypothesis 3 (**$H_3$**):** Local AI juries will show a higher variance in false-positive rates depending on model size (e.g., 8B parameters vs. 70B parameters) and quantization levels, with smaller models exhibiting lower adherence to complex legal standards.
    

## 2. Refined Multi-Agent System Architecture

To prevent sycophancy, maximize local compute efficiency, and simulate realistic human cognitive variance, we propose a **Two-Stage Multi-Agent Workflow** incorporating a **Randomized Persona Databank**:

```
[ Trial Transcript Input ]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: INDEPENDENT ANALYSIS (Secret Ballot)          │
│ - 12 Juror Agents read transcript independently.        │
│ - System prompts injected with distinct raw Big Five    │
│   (OCEAN) scores drawn from the Persona Databank.      │
│ - Each writes a private "Pre-Deliberation Verdict Memo".│
└────────────────────────┬───────────────────────────────┘
                         │ (Memos submitted to Master)
                         ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: STRUCTURED DELIBERATION                       │
│ - Groupchat Master summarizes conflict points.         │
│ - Jurors debate ONLY the summarized points (3 rounds).  │
│ - Final anonymous vote taken.                          │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
                  [ Final Verdict ]

```

### The Juror Persona Databank

To prevent homogenous, sterile responses, we maintain an offline databank of $N \ge 100$ juror profiles. Before running a trial simulation, the system randomly selects 12 distinct profiles to assign to the juror agents.

Each profile contains a randomized, scientifically validated Big Five (OCEAN) personality profile where traits are submitted directly as integer values ($[1, 100]$) to prevent LLM narrative hallucination or phrasing bias:

-   **Openness (**$O$**):** $1 \text{ to } 100$ (Circumstantial logic processing)
    
-   **Conscientiousness (**$C$**):** $1 \text{ to } 100$ (Adherence to judicial rules and legal scripts)
    
-   **Extraversion (**$E$**):** $1 \text{ to } 100$ (Vocal assertiveness and speaking frequency in debate)
    
-   **Agreeableness (**$A$**):** $1 \text{ to } 100$ (Susceptibility to consensus and peer pressure)
    
-   **Neuroticism (**$N$**):** $1 \text{ to } 100$ (Emotional sensitivity vs. logical detachment)
    

## 3. Recommended Prompts

### A. Base Juror System Prompt (The Blackstone Standard)

```
Role: You are a member of a 12-person criminal jury. Your civic duty is to analyze the provided trial transcript and determine if the state has proven the defendant's guilt beyond a reasonable doubt.

Core Directive (The Presumption of Innocence):
You must adhere to the fundamental legal standard: "It is better that ten guilty persons escape than that one innocent suffer." The defendant is presumed innocent. If there is a logical, reasonable explanation of the facts that points to innocence, you MUST vote NOT GUILTY. 

Your cognitive and decision-making framework is strictly governed by the following scientifically validated Big Five (OCEAN) personality scores, measured on a scale of 1 to 100 (where 1 is extremely low and 100 is extremely high):

[TRAIT LEGEND]
- Openness: {openness}/100 (Your capacity to parse complex forensic theories and alternative alibis)
- Conscientiousness: {conscientiousness}/100 (Your strict adherence to legal scripts and judicial rules)
- Extraversion: {extraversion}/100 (Your vocal assertiveness and willingness to initiate arguments in debate)
- Agreeableness: {agreeableness}/100 (Your value of social harmony, compromise, and susceptibility to peer pressure)
- Neuroticism: {neuroticism}/100 (Your emotional sensitivity to criminal details vs. cool logical detachment)

[YOUR ACTIVE PROFILE]
- Juror ID: {juror_id}
- Openness: {openness}
- Conscientiousness: {conscientiousness}
- Extraversion: {extraversion}
- Agreeableness: {agreeableness}
- Neuroticism: {neuroticism}

[INSTRUCTIONS]
Evaluate the evidence and trial scripts strictly through the lens of this psychological profile. Do not break character. Do not list your numeric scores in conversation; simply allow them to naturally govern how you analyze the data, formulate your secret ballot, and debate in the group chat.

```

### B. Groupchat Master / Orchestrator Prompt

```
Role: You are the Court Foreperson / Jury Facilitator. Your task is to lead 12 AI Jurors to a unanimous consensus.

Instructions:
1. Collect the initial private verdicts from all 12 jurors.
2. If there is 12-0 unanimity, declare the verdict and end the simulation.
3. If there is a split, identify the primary points of logical conflict (e.g., eyewitness credibility vs. circumstantial evidence).
4. Present these conflict points to the jury and moderate a structured, 3-round debate. Prevent circular arguments.
5. After Round 3, call for a final private vote.

```

## 4. Academic Paper Structure

### Title Ideas

-   _Beyond Reasonable Doubt in the Silicon Courtroom: Measuring False Positive Rates in Multi-Agent LLM Juries_
    
-   _The Silicon Jury Box: Exploring Persona Diversity and Error Rates in Multi-Agent Legal Consensus_
    

### I. Introduction

-   **Context:** The rise of open-source local LLM agents as proxies for human cohorts ("silicon sampling") to protect sensitive trial data.
    
-   **The Problem:** The legal system is built on specific cognitive standards (e.g., Blackstone’s Ratio, Beyond a Reasonable Doubt). We do not know if open-source models structurally emulate these safety parameters locally or if they default to optimization biases that inflate false positives.
    
-   **Contribution:** This paper presents a systematic, blinded framework comparing AI jury errors to human jury errors across identical evidentiary datasets, utilizing a randomized, high-variance persona architecture to emulate human cognitive diversity on local hardware.
    

### II. Literature Review

-   **Multi-Agent Dynamics:** Reviewing the limits of group deliberation in LLMs (sycophancy, confidence drift, cascade effects).
    
-   **Silicon Sampling:** How LLMs simulate human cohorts in social science.
    
-   **Cognitive Diversity in LLMs:** Why homogenous prompts fail to simulate real-world crowds, and how persona generation introduces necessary variance.
    
-   **Open-Source Local Inference in High-Stakes Legal Domains:** The trade-offs between cloud API models and local, zero-cost, private inference (Llama-3, Gemma-2).
    

### III. Methodology & Experimental Design

-   **Data Sources:** Selection of historical trials from databases like the _National Registry of Exonerations_ (representing proven false positives/wrongful convictions) versus standard convictions.
    
-   **Persona Databank Synthesis:** How the database of $N$ distinct juror personas was generated, categorized, and mathematically validated for behavioral variance.
    
-   **Agent Architecture:** Detail the Multi-Agent Groupchat Master and the 12-agent "Secret Ballot to Deliberation" pipeline over local Ollama endpoints.
    
-   **Statistical Analysis:** Formulae used to compute and compare False Positive Rates ($FPR$) and False Negative Rates ($FNR$):
    
    $$FPR = \frac{FP}{FP + TN}$$$$FNR = \frac{FN}{FN + TP}$$

### IV. Results

-   **Quantitative Verdict Comparisons:** Did local AI juries convict more or less often than human counterparts?
    
-   **The Persona Variance Effect:** Did the introduction of raw OCEAN values lower the homogeneous "herd consensus" rate in Phase 1? Which persona classes were most correlated with false-positive outcomes?
    
-   **Local Compute Influence:** Did model parameter size (8B vs. 70B) statistically impact the False Positive Rate ($FPR$)?
    
-   **The Deliberation Drift:** Did deliberation decrease or increase false-positive rates? Did minority dissenting personas successfully sway the majority?
    

### V. Discussion

-   **Aligning AI with Legal Philosophy:** Can local software actually comprehend "reasonableness" or does it merely calculate linguistic likelihood?
    
-   **Privacy & Ethical Implications:** The benefits of local, zero-cost inference pipelines for sensitive legal analysis and mock trials.
    
-   **Limitations:** Context window constraints on smaller local models, hardware bottlenecks, lack of non-verbal cues (witness demeanor) in textual transcripts.
    

### VI. Conclusion & Future Directions

-   Summary of findings.
    
-   Recommendations for optimizing multi-agent local consensus algorithms for high-stakes decision-making.
