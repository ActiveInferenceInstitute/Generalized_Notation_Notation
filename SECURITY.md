# Security Policy

> **📋 Document Metadata**  
> **Type**: Security Policy | **Audience**: All Users | **Complexity**: Intermediate  
> **Last Updated**: 2026-09-07 | **Status**: Maintained  
> **Cross-References**: [Comprehensive Security Guide](docs/security/README.md) | [Deployment Security](docs/deployment/README.md) | [MCP Security](docs/mcp/README.md)

## 🔒 Comprehensive Security Framework

The GNN (GeneralizedNotationNotation) project maintains a comprehensive multi-layered security approach covering development, deployment, and production environments.

> **📖 Complete Security Documentation**: For comprehensive security information, see [Security Guide](docs/security/README.md)

## Supported Versions

We are committed to ensuring the security of the GeneralizedNotationNotation (GNN) project.

| Version | Supported | Security Coverage |
| ------- | ------------------ | ----------------- |
| 4.0.0   | ✅ Full support | Current-run identity and bounded execution contracts |
| 3.6.0   | ✅ Full support | Complete security framework |
| 3.5.0   | ✅ Full support | Complete security framework |
| 3.3.0   | ✅ Full support | Complete security framework |
| 3.2.x   | ✅ Full support | Complete security framework |
| 3.1.x   | ✅ Full support | Complete security framework |
| 3.0.x   | ✅ Security fixes | Complete security framework |
| 2.0.x   | ❌ EOL | No security support |
| 1.x     | ❌ EOL | No security support |
| < 1.0.0 | ❌ Unsupported | No security support |

> **🔄 Version Updates**: This table is updated with each release. See [CHANGELOG.md](CHANGELOG.md) for version history.

## 📋 Recent CVE Remediation History

| Date | CVE ID | Package | Action |
|------|--------|---------|--------|
| 2026-01-27 | CVE-2026-24486 | python-multipart | Upgraded 0.0.21 → 0.0.22 |
| 2026-02-06 | CVE-2026-0994 | protobuf | Upgraded 6.33.4 → 6.33.5 (patched) |
| 2026-02-09 | CVE-2025-53000 | nbconvert | Upgraded 7.16.6 → 7.17.0 |
| 2026-02-11 | CVE-2026-26007 | cryptography | Upgraded 46.0.3 → 46.0.5 |
| 2026-03-05 | CVE-2025-14009 | nltk | Upgraded 3.9.2 → 3.9.3 (Zip Slip RCE fix) |
| 2026-03-05 | CVE-2026-28802 | authlib | Upgraded 1.6.6 → 1.6.9 (alg:none signature bypass fix) |
| 2026-03-22 | GHSA-rf74-v2fm-23pw, CVE-2026-33230, CVE-2026-33231 | nltk | Removed `safety` dev tool (sole lockfile consumer); PyPI has no release newer than 3.9.3 yet — monitor [nltk](https://pypi.org/project/nltk/) |
| 2026-05-07 | CVE-2026-4539 | pygments | Replaced third-party git fork override with official PyPI release ≥ 2.20.0 (AdlLexer ReDoS fix) |
| 2026-05-07 | CVE-2026-40192 | Pillow | Verified locked at 12.2.0 (FITS decompression bomb fix) |
| 2026-05-07 | CVE-2026-25990 | Pillow | Verified locked at 12.2.0 (PSD out-of-bounds write fix; patched in 12.1.1) |
| 2026-05-07 | CVE-2026-22815 | aiohttp | Verified locked at 3.13.5; tightened floor to ≥ 3.13.4 (header/trailer DoS) |
| 2026-05-07 | CVE-2026-21441 | urllib3 | Verified locked at 2.6.3 (decompression bomb bypass fix) |
| 2026-05-07 | CVE-2026-24049 | setuptools/wheel | Accepted risk — `wheel.unpack` path traversal not reachable in GNN workflows (see below) |
| 2026-06-24 | GHSA-6v7p-g79w-8964 | msgpack | Upgraded 1.1.2 → 1.2.1 (Unpacker reuse out-of-bounds read/crash fix) |
| 2026-06-24 | GHSA-vmhf-c436-hxj4 | jupyterlab | Upgraded 4.5.8 → 4.6.0 (extension manager stored XSS fix; patched in 4.5.9) |
| 2026-06-24 | GHSA-fcw5-x6j4-ccmp, CVE-2026-44727 | jupyter-server | Upgraded 2.19.0 → 2.20.0 (`Nbconvert*Handler` sandbox CSP fix) |
| 2026-06-24 | GHSA-gj48-438w-jh9v, GHSA-g75f-g53v-794x, GHSA-8rfp-98v4-mmr6 | bleach | Upgraded 6.3.0 → 6.4.0 (URI sanitization and linkify CPU exhaustion fixes) |
| 2026-09-07 | (floors, not CVEs) | numpy, pandas, openai, pytest, mypy | Raised declared floors toward the locked generation (numpy ≥ 2.1.0, pandas ≥ 2.0, openai ≥ 1.0, pytest ≥ 8.0, mypy ≥ 1.0); `uv lock` re-resolved with zero pin changes. Closes the stale-floor drift documented in `CHANGELOG.md` [Unreleased]. |

> **ℹ️ Known Accepted Risks**: The following vulnerabilities are documented and accepted:
>
> - **CVE-2024-39236** (gradio): Disputed — self-attack scenario only.
> - **CVE-2026-24049** (setuptools / bundled wheel): Path traversal in `wheel.unpack()`. `setuptools` 81.0.0 vendors wheel code that may be affected. Mitigation: GNN never calls `wheel unpack` or processes untrusted `.whl` archives at runtime; this is a build-tool-only surface. Monitor for a setuptools release that bundles `wheel >= 0.46.2`.
> - ~~**CVE-2022-42969** / **PYSEC-2022-42969** (`py` 1.11.0)~~: Resolved — `py` package is no longer present in the dependency graph (removed from `uv.lock` as of 2026-05-07).

## Recent Internal Security Remediation

| Date | Assessment | Scope | Outcome |
|------|------------|-------|---------|
| 2026-08-14 | Red-team review (historical `RED_TEAM_REVIEW.md` ledger) | Rendered-code execution ordering, pickle/literal-eval input parsing, FastAPI/MCP auth + path traversal, error disclosure, sandboxing | Closed V-01/V-03/V-04/V-05/V-06/V-07/V-09/V-10: pre-execution AST gate before Step 12, bounded `safe_literal_eval`, optional `GNN_API_KEY` auth + secure-bind refusal, symlink rejection, MCP forwarded-identity rejection, stderr path redaction, and an opt-in `GNN_SANDBOX` wrapper. The remaining items from the same wave were closed afterwards — see `CHANGELOG.md` §"Security (2026-08-14, wave 2 — residual closures)". |
| 2026-06-24 | Codex Security standard scan | MCP execution, MCP LLM file access, generated bnlearn code, generated artifact paths | Closed four reportable findings with repository-local MCP path validation, Step 11 render-summary execution gating, generated-code literal escaping, safe output filename stems, and regression tests. See [Codex Security Remediation - 2026-06-24](docs/security/codex_security_remediation_2026-06-24.md). |

## Dependency automation and local audits

- **Dependabot**: Version updates are configured in [`.github/dependabot.yml`](.github/dependabot.yml) (Python/`uv.lock` and GitHub Actions). Review alerts under **GitHub → Security → Dependabot**.
- **PRs**: [`.github/workflows/dependency-review.yml`](.github/workflows/dependency-review.yml) runs on pull requests to `main`.
- **Local check** (core runtime graph, no project package, no hashes):

  ```bash
  uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/requirements-audit.txt
  uv tool run pip-audit -r /tmp/requirements-audit.txt
  ```

- **Scheduled visibility**: [`.github/workflows/supply-chain-audit.yml`](.github/workflows/supply-chain-audit.yml) runs `pip-audit` weekly (non-blocking).

## 🚨 Reporting Security Vulnerabilities

The GNN team and community take all security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### Reporting Channels

**Primary Contact:**

- **Email**: Send an email to `blanket@activeinference.institute`
- **Subject Line**: Use "Security Vulnerability in GNN Project"

**GitHub Security:**

- **Platform**: [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
- **Repository**: [GeneralizedNotationNotation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/security/advisories)
- **Benefits**: Automated coordination with dependency maintainers

> **⚠️ Important**: Please do not report security vulnerabilities through public GitHub issues.

### What to Include

When reporting a vulnerability, please provide:

- **Clear description** of the vulnerability and its impact
- **Component identification**: Affected files, modules, or pipeline steps
- **Reproduction steps**: Detailed steps to reproduce the issue  
- **Version information**: Affected GNN versions and dependencies
- **Environment details**: Operating system, Python version, framework versions
- **Proof of concept**: If applicable, demonstration code (safely)
- **Suggested mitigations**: If you have ideas for fixes

### Security-Specific Concerns

**LLM Integration Security** (Pipeline Step 13):

- API key exposure in configuration files
- Prompt injection attacks through GNN files  
- Unsafe code generation from LLM outputs

**MCP Security** (Pipeline Step 21):

- Model Context Protocol authentication issues
- Unsafe resource access patterns
- Data leakage through model context

**Pipeline Security** (All 25 Steps):

- Code injection through GNN file parsing
- Unsafe file operations in output generation
- Privilege escalation in execution steps

## 🛡️ Our Security Commitment

### Response Timeline

Once a security vulnerability is reported, we commit to:

**Immediate Response (24-48 hours):**

- Acknowledge receipt of the vulnerability report
- Assign a security team member as primary contact
- Begin initial assessment and triage

**Investigation Phase (1-7 days):**

- Validate and reproduce the vulnerability
- Assess severity using CVSS scoring
- Determine affected versions and components
- Develop initial mitigation strategies

**Resolution Phase (Variable, based on severity):**

- **Critical**: 24-72 hours for emergency patch
- **High**: 1-2 weeks for comprehensive fix
- **Medium**: 2-4 weeks for scheduled release
- **Low**: Next planned release cycle

**Disclosure Phase:**

- Coordinate responsible disclosure timeline
- Prepare security advisory and documentation
- Release patched versions across supported branches
- Publicly acknowledge contributor (unless requested otherwise)

### Security Integration

**Development Security:**

- All code changes reviewed for security implications
- Automated security scanning in CI/CD pipeline  
- Dependency vulnerability monitoring
- Regular security audits of critical components

**Documentation Security:**

- Security considerations in all operational guides
- Threat model documentation for each pipeline step
- Security configuration examples and best practices
- Incident response procedures and playbooks

## 🔧 Security Best Practices for Users

### Development Environment Security

**Environment Setup:**

- Use isolated Python virtual environments
- Keep dependencies updated: `uv sync --refresh`
- Validate GNN file sources before processing
- Use secure API key storage (environment variables, not files)

**Code Security:**

- Review generated code before execution
- Validate all inputs to GNN parsers
- Use sandbox environments for testing unknown models
- Follow secure coding practices for extensions

### Production Deployment Security

**Infrastructure Security:**

- Deploy with minimal required privileges
- Use encrypted connections for all API calls
- Implement proper logging and monitoring
- Regular security updates and patches

**Configuration Security:**

- Secure API key management (Azure Key Vault, AWS Secrets Manager)
- Network segmentation for GNN processing
- Input validation for all user-provided GNN files
- Output sanitization for generated code

**Built-in security controls (env-var opt-in, 2026-08-14):**

- `GNN_API_KEY` — require a matching `X-API-Key` header on the FastAPI API;
  non-loopback binds are refused unless set or `GNN_ALLOW_INSECURE_BIND=1`.
- `GNN_SANDBOX` — `off` / `prefer` / `require` wrapping of rendered-script
  execution under firejail / bubblewrap / nsjail (degrades loudly, never silently).
- `GNN_ALLOW_UNSAFE_EXEC=1` — explicit escape hatch to bypass the
  pre-execution AST gate (trusted-local research only).
- `GNN_MCP_TOKEN` / `GNN_MCP_RATE_LIMIT_PER_MINUTE` — bearer auth and
  per-client rate limiting for the MCP HTTP server.

### Repository-local API filesystem scope

The local API resolves request paths inside a trusted, locally managed checkout.
Its shared path validator checks lexical containment, rejects existing symlink
components, and checks resolved containment before directory creation or returning
paths to the request handlers. Rejected paths produce a validation error before
backend processing. These checks constrain request strings; they do not provide
descriptor-based, atomic confinement against a separate actor replacing filesystem
entries between validation and use. Authentication and non-loopback binding controls
remain separate from this filesystem limitation. Use a checkout whose filesystem
writers are trusted; hostile concurrent filesystem mutation is outside the current
API contract.

### Framework-Specific Security

**PyMDP Security:**

- Validate matrix dimensions before processing
- Sanitize numerical inputs for stability
- Monitor memory usage for large state spaces

**RxInfer.jl Security:**  

- Validate Julia code generation outputs
- Secure inter-process communication with Julia
- Monitor computational resource usage

**ActiveInference.jl Security:**

- Validate Julia ActiveInference.jl code generation outputs
- Secure inter-process communication with Julia
- Monitor computational resource usage for ActiveInference.jl simulations

**LLM Integration Security:**

- Never include sensitive data in prompts
- Validate all LLM-generated outputs
- Use prompt injection prevention techniques
- Implement rate limiting for API calls

## 📚 Security Resources

### Comprehensive Guides

- **[Complete Security Framework](docs/security/README.md)** - Comprehensive security guide
- **[Deployment Security](docs/deployment/README.md)** - Production security configurations
- **[MCP Security](docs/mcp/README.md)** - Model Context Protocol security measures

### Framework Security

- **[PyMDP Security](docs/pymdp/gnn_pymdp.md#security-considerations)** - PyMDP-specific security
- **[RxInfer.jl Security](docs/rxinfer/gnn_rxinfer.md#security-considerations)** - Julia integration security
- **[ActiveInference.jl Security](docs/activeinference_jl/activeinference-jl.md#security-considerations)** - ActiveInference.jl integration security
- **[LLM Security](docs/llm/security_guidelines.md)** - AI integration security practices

### Incident Response

- **[Security Incident Response](docs/security/incident_response.md)** - Response procedures
- **[Vulnerability Assessment](docs/security/vulnerability_assessment.md)** - Assessment frameworks
- **[Security Monitoring](docs/security/monitoring.md)** - Monitoring and alerting

## 🤝 Security Community

### Contributing to Security

- **Security Review**: Participate in security-focused code reviews
- **Vulnerability Research**: Help identify potential security issues
- **Documentation**: Improve security documentation and guides
- **Tool Development**: Create security-focused tools and utilities

### Security Updates

- **Security Announcements**: Subscribe to repository notifications
- **Release Notes**: Check [CHANGELOG.md](CHANGELOG.md) for security fixes
- **Community Forum**: Engage in security discussions
- **Best Practices**: Share security configurations and patterns

---

**We appreciate your help in keeping GeneralizedNotationNotation secure across all dimensions: physical, digital, and cognitive.**

> **🔗 Related Documentation**: [Security Guide](docs/security/README.md) | [Deployment Security](docs/deployment/README.md) | [Contributing Security](CONTRIBUTING.md#security-considerations)
