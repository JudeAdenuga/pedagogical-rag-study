# Pedagogical RAG Study

## Retrieval-Augmented Generation for Pedagogically Aware Educational AI  
### A Comparison of a Baseline LLM Tutor and a Learner-State-Aware Pedagogical RAG Tutor

This repository contains the implementation and evaluation materials for a controlled educational AI study examining whether retrieval-augmented generation, learner-state tracking, and bounded pedagogical response policies can improve the instructional quality of large language model tutoring in algebra.

The study compares two instructional conditions:

1. **Baseline LLM Tutor**  
   A fixed, prompt-only, stateless tutoring condition implemented with `gemini-2.5-flash`.

2. **Pedagogical RAG Tutor**  
   A learner-state-aware tutoring condition implemented with the same `gemini-2.5-flash` foundation model, augmented with:
   - curriculum-grounded retrieval,
   - structured learner-state tracking,
   - deterministic pedagogical response-control logic, and
   - scaffolded tutoring modes such as guided questioning, adaptive hinting, misconception correction, and worked-step support.

By holding the foundation model constant across both instructional conditions, the study is designed to reduce model-family confounding and more directly examine the contribution of retrieval-grounded pedagogical augmentation.

---

## Research Purpose

Large language models can generate fluent tutoring responses, but fluency alone does not guarantee reliable educational support. In mathematics tutoring, especially across multi-turn interactions, a useful system must do more than produce a correct answer. It should maintain instructional continuity, respond to the learner’s apparent state of understanding, provide appropriately calibrated help, and reduce unsupported or pedagogically weak explanations.

This project investigates whether a pedagogically structured RAG tutor can provide stronger educational support than a prompt-only baseline LLM tutor under controlled conditions.

The broader aim is not to position AI as a replacement for teachers, but to explore how educational AI systems may be designed more responsibly as instructional support tools grounded in curriculum evidence and pedagogical reasoning.

---

## Study Design

### Overall Design

The study uses a **two-condition comparative evaluation** in the domain of algebra tutoring:

| Condition | Description |
|---|---|
| **Baseline LLM Tutor** | `gemini-2.5-flash` used in a fixed prompt-only, stateless configuration with no retrieval, no external instructional evidence, and no learner-state memory. |
| **Pedagogical RAG Tutor** | `gemini-2.5-flash` augmented with retrieved instructional evidence, learner-state variables, and bounded pedagogical policy logic that guides the style and level of tutoring support. |

Both conditions are evaluated on the same benchmark tasks and interaction scenarios so that differences in output quality can be attributed as directly as possible to the pedagogical augmentation layer rather than to differences in the underlying language model.

---

## Instructional Condition 1: Baseline LLM Tutor

The baseline tutor serves as the study’s comparison condition. It is intentionally designed as a **clean, prompt-only evaluation harness**, rather than as a production tutoring application.

### Baseline Characteristics

- Model: `gemini-2.5-flash`
- Fixed tutoring prompt
- Stateless across turns
- No retrieval
- No uploaded instructional corpus
- No learner-state tracking
- No pedagogical policy engine
- No external grounding or evidence injection

The baseline is included to represent a common form of LLM tutoring support: a capable general-purpose model responding directly to student-facing algebra prompts without structured educational augmentation.

---

## Instructional Condition 2: Pedagogical RAG Tutor

The pedagogical RAG tutor uses the same underlying model, `gemini-2.5-flash`, but adds three core layers of educational control:

1. **Curriculum-Grounded Retrieval**  
   Relevant instructional evidence is retrieved from a curated algebra corpus designed to support concept explanations, worked examples, misconception handling, and stepwise problem solving.

2. **Learner-State Tracking**  
   The system maintains structured interaction variables outside the model, including information such as:
   - problem identifier,
   - algebra topic and subskill,
   - recent learner attempt,
   - detected error type,
   - number of hints already provided,
   - current support level,
   - rolling interaction summary, and
   - last pedagogical response mode.

3. **Pedagogical Policy Logic**  
   A deterministic policy layer selects an appropriate tutoring mode based on the learner state and task context. These bounded response modes include:
   - conceptual prompt,
   - light hint,
   - procedural hint,
   - misconception correction,
   - worked-step explanation, and
   - full solution when warranted.

### Purpose of the Pedagogical RAG Condition

The RAG condition is designed to test whether an LLM tutor becomes more instructionally effective when it is:
- grounded in task-relevant educational evidence,
- informed by the ongoing learner interaction,
- and constrained by explicit pedagogical response rules rather than relying only on unconstrained generative behavior.

---

## Evaluation Domain

The study focuses on **algebra tutoring** because algebra provides a useful testbed for evaluating both:
- procedural correctness, and
- pedagogical quality across multi-turn instructional interactions.

The benchmark includes **24 algebra tasks** distributed across several instructional categories, including:
- simplifying expressions,
- solving linear equations,
- substitution,
- error identification,
- symbolic relationship reasoning, and
- conceptual algebra prompts.

These tasks are used to generate matched interactions across both tutoring conditions.

---

## Evaluation Approach

The study evaluates system outputs using a structured expert-review framework rather than live student deployment in the current phase.

### Primary Evaluation Dimensions

Outputs are assessed across several dimensions:

- **Solution Accuracy**  
  Whether the mathematical response is correct.

- **Conceptual Support**  
  Whether the tutor helps the learner understand the underlying idea rather than only giving an answer.

- **Instructional Quality**  
  Whether the response is clear, appropriately scaffolded, and pedagogically useful.

- **Hallucination or Unsupported Content**  
  Whether the system introduces incorrect, irrelevant, or ungrounded claims.

- **Expert-Rated Instructional Helpfulness**  
  Whether the response would plausibly support a learner’s progress in a tutoring context.

### Evaluation Structure

The project uses:
- a controlled task bank,
- matched prompt scenarios across both conditions,
- multi-turn tutoring episodes,
- structured output logging,
- rubric-based expert scoring, and
- comparison of baseline versus pedagogical RAG responses.

No human-subject classroom trial is conducted in the present phase. The study is currently focused on implementation, transcript generation, and expert evaluation of system behavior.

---

## System Logic

At a high level, the pedagogical RAG tutoring pipeline follows this sequence:

1. Receive learner task or conversational turn  
2. Parse the task context and recent interaction state  
3. Retrieve curriculum-relevant instructional evidence  
4. Update learner-state variables  
5. Select a pedagogical response mode using deterministic policy rules  
6. Construct the tutoring prompt  
7. Generate the response using `gemini-2.5-flash`  
8. Log the episode for later analysis and evaluation

The baseline system uses a reduced pipeline:

1. Receive learner task or conversational turn  
2. Construct the fixed baseline tutoring prompt  
3. Generate the response using `gemini-2.5-flash`  
4. Log the episode for later comparison

---

## Repository Scope

This repository is intended to support the full study workflow, including:

- benchmark task management,
- baseline tutoring prompt construction,
- pedagogical RAG prompt construction,
- learner-state schema and update logic,
- deterministic pedagogical policy rules,
- retrieval-grounded tutoring orchestration,
- matched evaluation runs,
- structured logging of outputs,
- and preparation of evidence for expert scoring and manuscript reporting.

---

## Study Contribution

This project contributes to educational AI development by examining a specific design question:

> Does a learner-state-aware, retrieval-grounded tutoring system provide stronger pedagogical support than a prompt-only LLM tutor when the underlying foundation model is held constant?

The study is positioned at the intersection of:
- retrieval-augmented generation,
- intelligent tutoring systems,
- pedagogically aware AI design,
- learner-state modeling,
- and responsible educational applications of large language models.

Rather than treating tutoring quality as a matter of response fluency alone, the project evaluates whether instructional systems can be designed to better support reasoning, conceptual development, and calibrated assistance.

---

## Current Project Phase

The project is currently in the implementation and evaluation phase. Core study materials include:

- the 24-task algebra benchmark,
- baseline prompt specification,
- pedagogical RAG prompt specification,
- learner-state schema,
- pedagogical policy rules,
- retrieval corpus samples,
- structured scoring rubric,
- and output logs for matched baseline versus pedagogical RAG comparisons.

These materials support both the technical build and the accompanying scholarly manuscript.

---

## Author

**Jude A. Adenuga**  
Educational Technology and Instructional Design  
Research focus: educational AI, retrieval-augmented generation, learner-state-aware tutoring systems, and pedagogically grounded LLM applications.

---

## Project Title for Manuscript

**Retrieval-Augmented Generation for Pedagogically Aware Educational AI: A Comparison of Baseline LLMs and Learner-State-Aware RAG**