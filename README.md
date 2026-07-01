# Pedagogical RAG Study

## Retrieval-Augmented Generation for Pedagogically Aware Educational AI  
### An Expert-Rated Comparison of a Prompt-Only LLM Tutor and an Integrated Learner-State-Aware RAG Tutor

This repository contains implementation and evaluation materials for a controlled educational AI study in algebra tutoring.

The study compares two instructional conditions:

1. **Prompt-Only LLM Tutor**  
   A fixed, stateless tutoring condition implemented with `gemini-2.5-flash`.

2. **Integrated Learner-State-Aware RAG Tutor**  
   A tutoring condition implemented with the same `gemini-2.5-flash` foundation model, augmented with:
   - curriculum-grounded retrieval,
   - structured learner-state tracking,
   - deterministic pedagogical response-control logic, and
   - scaffolded tutoring modes such as guided questioning, adaptive hinting, misconception correction, and worked-step support.

The foundation model is held constant across both conditions. This allows the study to examine output differences associated with the integrated pedagogical augmentation layer.

This repository is intended for research inspection and reproducibility. It is not a public tutoring product.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/JudeAdenuga/pedagogical-rag-study.git
   cd pedagogical-rag-study
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables.

   Create a `.env` file in the repository root using `.env.example` as a guide.

   Example:

   ```env
   GEMINI_API_KEY=your_api_key_here
   MODEL_NAME=gemini-2.5-flash
   ```

   Do not commit real API keys or private credentials to the repository.

---

## How to Use

This repository supports the controlled comparison reported in the manuscript.

1. Review the benchmark task materials in `manuscript_artifacts/`.
2. Review the baseline and pedagogical RAG prompt templates.
3. Configure the required environment variables using `.env.example`.
4. Run the prompt-only baseline workflow to generate baseline tutor outputs.
5. Run the pedagogical RAG workflow to generate retrieval-grounded, learner-state-aware outputs.
6. Export the logged outputs for expert scoring and analysis.
7. Compare the outputs using the rubric and analysis materials provided with the study artifacts.

The workflow is designed for research replication and inspection. It should not be used as a live student-facing tutoring system without additional safety testing, output filtering, and classroom validation.

---

## Research Purpose

Large language models can generate fluent tutoring responses, but fluency alone does not guarantee reliable educational support. In mathematics tutoring, a useful system must do more than produce a correct answer. It should maintain instructional continuity, respond to the learner’s apparent state of understanding, provide appropriately calibrated help, and avoid unsupported or misleading explanations.

This project examines whether an integrated learner-state-aware RAG tutor produces stronger expert-rated tutoring responses than a prompt-only LLM tutor under controlled conditions.

The study does not claim to measure student learning gains. No live classroom deployment was conducted in the present phase. The current work focuses on system implementation, transcript generation, and expert evaluation of tutor responses.

---

## Study Design

The study uses a two-condition comparative evaluation in the domain of algebra tutoring.

| Condition | Description |
|---|---|
| **Prompt-Only LLM Tutor** | `gemini-2.5-flash` used in a fixed, stateless prompt-only configuration with no retrieval, no external instructional evidence, and no learner-state memory. |
| **Integrated Learner-State-Aware RAG Tutor** | `gemini-2.5-flash` used with retrieved instructional evidence, learner-state variables, and bounded pedagogical policy logic that guides the style and level of tutoring support. |

Both conditions are evaluated on the same benchmark tasks and interaction scenarios. This allows output quality to be examined while holding the foundation model constant.

---

## Instructional Condition 1: Prompt-Only LLM Tutor

The prompt-only tutor serves as the study’s comparison condition. It is designed as a clean evaluation harness rather than as a production tutoring application.

### Baseline Characteristics

- Model: `gemini-2.5-flash`
- Fixed tutoring prompt
- Stateless across turns
- No retrieval
- No uploaded instructional corpus
- No learner-state tracking
- No pedagogical policy engine
- No external grounding or evidence injection

The baseline represents a common form of LLM tutoring support: a capable general-purpose model responding directly to algebra prompts without structured educational augmentation.

---

## Instructional Condition 2: Integrated Learner-State-Aware RAG Tutor

The pedagogical RAG tutor uses the same underlying model, `gemini-2.5-flash`, but adds three layers of instructional control.

### 1. Curriculum-Grounded Retrieval

Relevant instructional evidence is retrieved from a curated algebra corpus. The corpus supports concept explanations, worked examples, misconception handling, and stepwise problem solving.

### 2. Learner-State Tracking

The system maintains structured interaction variables outside the model, including:

- problem identifier,
- algebra topic and subskill,
- recent learner attempt,
- detected error type,
- number of hints already provided,
- current support level,
- rolling interaction summary, and
- last pedagogical response mode.

### 3. Pedagogical Policy Logic

A deterministic policy layer selects a tutoring mode based on the learner state and task context. The bounded response modes include:

- conceptual prompt,
- light hint,
- procedural hint,
- misconception correction,
- worked-step explanation, and
- full solution when warranted.

---

## Evaluation Domain

The study focuses on algebra tutoring because algebra allows both procedural correctness and instructional quality to be examined.

The benchmark includes 24 algebra tasks across these categories:

- simplifying expressions,
- solving linear equations,
- substitution,
- worked-solution error identification,
- symbolic relationship reasoning, and
- conceptual algebra prompts.

These tasks are used to generate matched interactions across both tutoring conditions.

---

## Evaluation Approach

The study evaluates system outputs through structured expert review rather than live student deployment.

### Primary Evaluation Dimensions

Outputs are assessed across five dimensions:

- **Solution Accuracy**  
  Whether the mathematical response is correct.

- **Conceptual Support**  
  Whether the tutor explains the underlying idea rather than only giving an answer.

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
- two-turn tutoring episodes,
- structured output logging,
- rubric-based expert scoring, and
- comparison of prompt-only and pedagogical RAG responses.

No human-subject classroom trial is conducted in the present phase.

---

## System Logic

At a high level, the integrated learner-state-aware RAG tutoring pipeline follows this sequence:

1. Receive learner task or conversational turn.
2. Parse the task context and recent interaction state.
3. Retrieve curriculum-relevant instructional evidence.
4. Update learner-state variables.
5. Select a pedagogical response mode using deterministic policy rules.
6. Construct the tutoring prompt.
7. Generate the response using `gemini-2.5-flash`.
8. Log the episode for analysis and evaluation.

The prompt-only baseline uses a reduced pipeline:

1. Receive learner task or conversational turn.
2. Construct the fixed baseline tutoring prompt.
3. Generate the response using `gemini-2.5-flash`.
4. Log the episode for comparison.

---

## Repository Scope

This repository supports the study workflow, including:

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

This project examines a focused design question:

> Does an integrated learner-state-aware, retrieval-grounded tutoring system receive stronger expert ratings than a prompt-only LLM tutor when the foundation model is held constant?

The study is positioned at the intersection of:

- retrieval-augmented generation,
- intelligent tutoring systems,
- pedagogically aware AI design,
- learner-state modeling, and
- responsible educational applications of large language models.

The contribution is an early architecture and expert-evaluation study. It does not claim to prove classroom effectiveness or student learning gains.

---

## Current Project Phase

The project is in the implementation and evaluation phase. Core study materials include:

- the 24-task algebra benchmark,
- baseline prompt specification,
- pedagogical RAG prompt specification,
- learner-state schema,
- pedagogical policy rules,
- retrieval corpus samples,
- structured scoring rubric, and
- output logs for matched prompt-only versus pedagogical RAG comparisons.

These materials support the technical workflow and the accompanying scholarly manuscript.

---

## Repository Notes

- The repository is intended for research reproducibility.
- The system is not a deployed educational product.
- The included outputs are for controlled evaluation only.
- Real API keys and private credentials should not be committed.
- Student-facing deployment would require additional safety review, content filtering, and classroom validation.

---

## Author

**Jude A. Adenuga**  
Educational Technology and Instructional Design  
Research focus: educational AI, retrieval-augmented generation, learner-state-aware tutoring systems, and pedagogically grounded LLM applications.

---

## Manuscript Title

**Retrieval-Augmented Generation for Pedagogically Aware Educational AI: An Expert-Rated Comparison of a Prompt-Only LLM Tutor and an Integrated Learner-State-Aware RAG Tutor**
