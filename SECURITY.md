# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability in LINAL, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue
2. Email security details to: **<develop@gorigami.xyz>**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (see below)

### Severity Levels

- **Critical**: Remote code execution, data corruption, authentication bypass
  - Fix timeline: 7 days
- **High**: Information disclosure, privilege escalation
  - Fix timeline: 30 days
- **Medium**: Denial of service, limited information disclosure
  - Fix timeline: 90 days
- **Low**: Minor issues, best practice violations
  - Fix timeline: Next release

---

## Security Considerations

### Current Security Posture

LINAL is designed as an **in-memory analytical engine** for trusted environments. Key security considerations:

#### 1. **Input Validation**

- **DSL Commands**: All DSL commands are validated before execution
  - Maximum command length: 16KB
  - Non-empty checks
  - Syntax validation
- **Server Requests**: HTTP requests are validated
  - Content-Type checking
  - Size limits enforced
  - Query timeout (30s) prevents resource exhaustion

#### 2. **Memory Safety**

- Built in Rust, providing memory safety guarantees
- No unsafe code blocks in critical paths
- Bounds checking on all array/tensor operations

#### 3. **Data Isolation**

- Multi-database support provides logical isolation
- Each database has isolated stores
- No cross-database data leakage

#### 4. **Persistence**

- File-based persistence (Parquet, JSON)
- No automatic encryption (users should encrypt filesystem if needed)
- Metadata stored in plain JSON

#### 5. **Network Security**

- HTTP server has no built-in authentication
- No TLS/HTTPS support (use reverse proxy for production)
- Server binds to `0.0.0.0` by default (consider firewall rules)

### Known Limitations

1. **No Authentication/Authorization**
   - Server API has no built-in auth
   - All users have full access
   - **Recommendation**: Use reverse proxy (nginx, Caddy) with authentication

2. **No Encryption**
   - Data stored in plain text (Parquet, JSON)
   - No encryption at rest
   - **Recommendation**: Encrypt filesystem or use encrypted storage

3. **No Rate Limiting**
   - Server has no rate limiting
   - Vulnerable to DoS via excessive requests
   - **Recommendation**: Use reverse proxy with rate limiting

4. **Query Timeout Only**
   - 30s timeout prevents infinite loops
   - No resource quotas (memory, CPU)
   - **Recommendation**: Use container limits (Docker, Kubernetes)

5. **File System Access**
   - Full read/write access to data directory
   - No sandboxing
   - **Recommendation**: Run with restricted filesystem permissions

---

## Security Best Practices

### For Users

1. **Network Security**
   - Don't expose server to public internet
   - Use firewall rules
   - Use reverse proxy with TLS

2. **Access Control**
   - Implement authentication at reverse proxy level
   - Use network isolation (VPN, private network)
   - Restrict filesystem permissions

3. **Data Protection**
   - Encrypt sensitive data at filesystem level
   - Use encrypted storage volumes
   - Regular backups

4. **Resource Limits**
   - Use container resource limits
   - Monitor memory usage
   - Set appropriate query timeouts

5. **Input Validation**
   - Validate DSL commands before execution
   - Sanitize user inputs
   - Use parameterized queries (when available)

### For Developers

1. **Code Review**
   - All code changes require review
   - Security-sensitive changes get extra scrutiny

2. **Dependencies**
   - Regular dependency updates
   - Monitor for security advisories
   - Use `cargo audit` to check for vulnerabilities

3. **Testing**
   - Test security-critical paths
   - Fuzz testing for parsers
   - Integration tests for security features

4. **Documentation**
   - Document security assumptions
   - Note known limitations
   - Provide security guidance

---

## Security Updates

Security updates will be released as:

- **Patch releases** (e.g., 0.1.5 → 0.1.6) for critical/high severity
- **Minor releases** (e.g., 0.1.x → 0.2.0) for medium/low severity

All security updates will be documented in:

- `CHANGELOG.md` (with security section)
- GitHub Security Advisories
- Release notes

---

## Dependency Security

We use `cargo audit` to check for known vulnerabilities in dependencies:

```bash
cargo install cargo-audit
cargo audit
```

Regular dependency updates help maintain security.

---

## Reporting Security Issues

**Email**: <develop@gorigami.xyz>

**PGP Key**: (if available)

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

---

## Acknowledgments

We thank security researchers who responsibly disclose vulnerabilities. Contributors will be acknowledged (with permission) in:

- Security advisories
- Release notes
- `CHANGELOG.md`

---

## Security Roadmap

Future security enhancements planned:

- [ ] Optional authentication for server API
- [ ] TLS/HTTPS support
- [ ] Rate limiting
- [ ] Resource quotas
- [ ] Audit logging
- [ ] Encryption at rest (optional)
- [ ] Sandboxing for untrusted code execution

---

**Last Updated**: 2025

For questions about security, contact: <develop@gorigami.xyz>
