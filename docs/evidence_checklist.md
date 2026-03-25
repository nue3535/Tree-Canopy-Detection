# Evidence Checklist (Submission Readiness)

Use this checklist before final submission, demo, and oral defense.

## A. Core Project Scope

- [ ] Problem statement is clearly documented.
- [ ] CV task and real-world motivation are clearly explained.
- [ ] Dataset source, size, and split protocol are documented.
- [ ] Any dataset capture requirement is clarified with coordinator expectation.

## B. Implementation

- [ ] Method 1 (`DeepLabV3+`) implemented and runnable.
- [ ] Method 2 (`SAM2`) implemented and runnable.
- [ ] GUI is operational (upload + prediction workflow).
- [ ] Backend API endpoints are working (`/api/health`, `/api/segment`).

## C. Comparison and Performance

- [ ] Both methods evaluated on the same split/protocol.
- [ ] Quantitative comparison table is filled (`docs/results_comparison.md`).
- [ ] Qualitative examples captured for both methods.
- [ ] Final method choice is justified with evidence.
- [ ] Improvement proposals are listed and grounded in results.

## D. Artifacts and Reproducibility

- [ ] Trained checkpoints are available and referenced.
- [ ] Environment and run commands are documented in `README.md`.
- [ ] Large generated outputs are excluded from git tracking.
- [ ] Code repository is clean and organized.

## E. Presentation / Oral Defense

- [ ] Slide deck prepared (problem, method, results, comparison, conclusion).
- [ ] GUI demo script prepared.
- [ ] Recorded video (if required) is prepared.
- [ ] All team members can explain contributions and design decisions.

## F. Suggested Attachment Set

- [ ] `docs/assignment_alignment_report.md`
- [ ] `docs/results_comparison.md`
- [ ] `docs/screenshots/` populated
- [ ] Link to code repository
