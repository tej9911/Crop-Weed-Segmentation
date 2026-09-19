---
name: "Full Repository Health Audit"
description: "Inspect the entire repository, verify whether the application works end to end, and produce a detailed evidence-based diagnosis and solution report."
argument-hint: "Optional focus, known symptom, or command to prioritize"
agent: "agent"
---

Perform a complete health audit of the current repository and application. Treat the repository as the source of truth and do not assume that the README, deployment configuration, or UI claims are correct.

## Audit scope

Inspect all tracked source files, configuration files, scripts, documentation, assets, model files, and dependency declarations. Follow the real execution paths from the user interface through the backend and model/inference layer. For this project, pay particular attention to:

- Application startup locally and through the provided scripts
- Python imports, dependency compatibility, environment assumptions, and missing files
- Flask routes, HTTP methods, status codes, request validation, error handling, CORS, and static-file serving
- TensorFlow/Keras model discovery, custom layers, input/output shapes, serialization, and fallback behavior
- Image validation, preprocessing, classification logic, confidence/probability correctness, and edge cases
- Frontend-to-backend endpoint names, payloads, response fields, loading/error states, and client-side fallback behavior
- Deployment configuration, port handling, path handling, and whether the selected platform can run the application as configured
- Documentation and startup instructions versus the actual repository contents
- Security, reliability, observability, performance, accessibility, and user-facing correctness

## Evidence and execution

1. Start by inventorying the repository and identifying the application entry points and supported run modes.
2. Read the relevant code before forming conclusions. Trace each important behavior to the code that controls it.
3. Run the cheapest relevant checks available, then progressively stronger checks: syntax/import checks, dependency or configuration checks, endpoint tests, and a real local smoke test where the environment permits it.
4. Exercise both success and failure paths, including missing/invalid uploads, unsupported image formats, empty files, unavailable model files, TensorFlow unavailable, malformed model output, and frontend/backend disconnects.
5. Do not hide or work around failures. If a check cannot run because of a missing dependency, model artifact, external service, or environment limitation, record the exact blocker and distinguish it from a code defect.
6. Do not modify repository files during the audit unless explicitly requested. The requested output is a report, not an unsolicited refactor.
7. Avoid calling an app “working” based only on static inspection. Separate verified behavior from inferred behavior.

## Required report

Return a detailed Markdown report with these sections, in this order:

### 1. Executive verdict
State one of: `Working`, `Working with material issues`, `Not working`, or `Not verifiable`. Give the short reason and list the exact checks that support the verdict.

### 2. What was checked
List the files/components inspected, commands run, runtime checks performed, and checks that were blocked. Include exit codes or concise output where useful.

### 3. Findings
For every issue, include:

- Severity: `Critical`, `High`, `Medium`, or `Low`
- Area: runtime, API, ML/model, frontend, deployment, security, documentation, or quality
- Exact file and line reference when available
- Observed behavior and reproducible evidence
- Expected behavior
- Root cause
- User impact and failure conditions
- Whether it is confirmed, probable, or unverified

Order findings by severity and avoid reporting style preferences as defects.

### 4. Earlier problem reconstruction
Explain the earlier problem as a technical incident reconstruction:

- What symptom a user would have seen
- Which code path produced it
- The earliest incorrect assumption or broken boundary
- Why the problem could have been missed by the existing checks or documentation
- Any contributing factors, such as missing artifacts, environment drift, inconsistent contracts, or silent fallbacks

If the earlier problem cannot be established from the repository or execution evidence, say so explicitly and present the most likely hypotheses separately.

### 5. Best solution
Recommend the smallest durable solution that fixes the root cause. Explain why it is preferable to superficial workarounds. Include:

- Specific files and code/configuration changes needed
- Required dependency, model, environment, or deployment changes
- API or data-contract changes, if any
- Validation and regression tests that should be added
- Any migration or rollout concerns

Do not present multiple options without choosing a recommended approach. If an issue is environmental rather than code-level, state the exact environment correction and the code guardrails still needed.

### 6. Verification plan
Give an ordered checklist of commands and manual checks that would prove the recommended solution works locally and in deployment. Include representative API requests and expected responses where relevant.

### 7. Residual risks
List remaining limitations, untested paths, assumptions, and risks that would prevent a claim of perfect operation.

Finish with a concise one-paragraph summary answering: “Is the app working perfectly, what was the earlier problem, and what is the best solution?”
