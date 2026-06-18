# Security Policy

This is a research scaffold, not a deployed service, so the attack surface is
small — but a few things are worth flagging.

## Reporting

Found something? Open a private security advisory via the repository's **Security
→ Advisories** tab, or email the maintainer rather than filing a public issue.

## Things to keep in mind when using this code

- **Untrusted audio / transcripts.** The data-prep code parses third-party TSVs
  and audio. Treat downloaded corpora as untrusted input; the manifest validators
  exist partly to catch malformed rows early.
- **Model weights.** Track A/B pull weights from the Hugging Face Hub. Pin
  revisions and verify checksums before running fine-tunes you intend to ship.
- **Licensing is a compliance issue, not just a legal one.** Several datasets and
  fine-tuned weights here are non-commercial (CC-BY-NC). See
  `docs/feasibility-report.md` for the per-asset breakdown before any commercial
  use.
