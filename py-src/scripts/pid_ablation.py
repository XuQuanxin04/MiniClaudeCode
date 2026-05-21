"""PID on/off ablation experiment.

Compares memory injection quality with PID control enabled vs disabled
across varying context pressure levels. Measures:
- Injection count (how many memories injected)
- Precision (are injected memories relevant?)
- Noise rate (cross-domain contamination)

This is the key experiment proving the cybernetic contribution.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time

sys.path.insert(0, ".")

from minicode.memory import MemoryEntry, MemoryFile, MemoryScope
from minicode.memory_injector import (
    MemoryInjectionController,
    MemoryInjectionSignal,
    MemoryInjector,
)
from minicode.domain_classifier import get_active_domain_values


# ── Build realistic memory dataset (same 80 as ablation) ──────────

ALL_MEMORIES = [
    ("fe-01", "React components must use functional style with hooks. No class components.", ["frontend"]),
    ("fe-02", "Forms use react-hook-form v7 with zod validation. Controller wrapper for third-party inputs.", ["frontend"]),
    ("fe-03", "CSS uses Tailwind v3 utility classes. Custom theme in tailwind.config.ts.", ["frontend"]),
    ("fe-04", "State management uses Zustand v4. Migrated from Redux Q1 2026.", ["frontend"]),
    ("fe-05", "Routing uses React Router v6 with lazy loading. Protected routes check auth.", ["frontend"]),
    ("fe-06", "API calls use axios with JWT interceptor. Base URL in src/api/client.ts.", ["frontend"]),
    ("fe-07", "All text must use i18next for i18n. Keys in src/locales/.", ["frontend"]),
    ("fe-08", "Button variants: primary blue, secondary gray, danger red.", ["frontend"]),
    ("fe-09", "Modal dialogs use @radix-ui/react-dialog. Always set aria-label.", ["frontend"]),
    ("fe-10", "Component tests use @testing-library/react + vitest. Target 80% coverage.", ["frontend", "testing"]),
    ("fe-11", "Date formatting uses date-fns v3. UTC internally, localize on display.", ["frontend"]),
    ("fe-12", "Image optimization: next/image with sizes attribute. Lazy load below fold.", ["frontend"]),
    ("fe-13", "Error boundaries wrap every route segment. Log errors to Sentry.", ["frontend"]),
    ("fe-14", "TypeScript strict mode enabled. No 'any' in new code.", ["frontend"]),
    ("fe-15", "ESLint extends @typescript-eslint/strict. Prettier formatting.", ["frontend"]),
    ("be-01", "API built with FastAPI 0.110. All endpoints use async handlers.", ["backend"]),
    ("be-02", "JWT authentication: access 15min, refresh 7 days. Rotate refresh tokens.", ["backend", "security"]),
    ("be-03", "Rate limiting: 100 req/min per user. Redis sliding window.", ["backend"]),
    ("be-04", "Input validation uses Pydantic v2 validators. Sanitize all input.", ["backend"]),
    ("be-05", "Error responses follow RFC 7807 Problem Details.", ["backend"]),
    ("be-06", "Logging uses structlog with JSON output. DEBUG dev, INFO staging, WARNING prod.", ["backend"]),
    ("be-07", "Background tasks use Celery with Redis broker. Max retry 3.", ["backend"]),
    ("be-08", "Email sending via Resend API. Queue via Celery, don't block.", ["backend"]),
    ("be-09", "File uploads to S3 via boto3. Max 10MB. Generate presigned URLs.", ["backend"]),
    ("be-10", "API versioning: /api/v1/ stable, /api/v2/ beta. Deprecation headers.", ["backend"]),
    ("be-11", "Webhooks use HMAC-SHA256 verification. Replay prevention.", ["backend", "security"]),
    ("be-12", "GraphQL at /graphql uses Strawberry. Query complexity limit 100.", ["backend"]),
    ("be-13", "Caching: Redis hot data 5min TTL, DB cold. Write-through pattern.", ["backend"]),
    ("be-14", "Pagination: cursor-based for lists, offset for search. Default 20.", ["backend"]),
    ("be-15", "Health check at /health: DB conn, Redis conn, Celery workers, last deploy.", ["backend"]),
    ("db-01", "Primary DB: PostgreSQL 16 on RDS. PostGIS extension. PgBouncer pooling.", ["database"]),
    ("db-02", "Migrations via Alembic. Never edit tables in production.", ["database"]),
    ("db-03", "Naming: tables=snake_case_plural, PK=id, FK=table_id. Indexes: idx_table_col.", ["database"]),
    ("db-04", "Query perf: EXPLAIN ANALYZE required for new table queries.", ["database"]),
    ("db-05", "JSONB for semi-structured data with GIN index. No EAV pattern.", ["database"]),
    ("db-06", "Read replicas: 2 for analytics. Write to primary only.", ["database"]),
    ("db-07", "Backup: daily snapshots 30 days. Point-in-time recovery. Monthly restore drill.", ["database"]),
    ("db-08", "Soft deletes: deleted_at timestamp, filter in app queries.", ["database"]),
    ("db-09", "Seed data in src/db/seeds/ for dev. Never seed production.", ["database"]),
    ("db-10", "Full-text search: PostgreSQL tsvector with GIN index.", ["database"]),
    ("do-01", "CI/CD: GitHub Actions. PR: lint+typecheck+test+build.", ["devops"]),
    ("do-02", "Docker: multi-stage builds. Base python:3.12-slim. Non-root user.", ["devops"]),
    ("do-03", "Environment config via .env files. Secrets in GitHub Secrets.", ["devops"]),
    ("do-04", "Kubernetes: EKS 1.29. HPA CPU>70% scales 2-10 pods.", ["devops"]),
    ("do-05", "Monitoring: Prometheus /metrics, Grafana dashboards. PagerDuty alert.", ["devops"]),
    ("do-06", "Log aggregation: Fluentd->OpenSearch. 30d hot, 90d warm.", ["devops"]),
    ("do-07", "SSL/TLS: cert-manager + Let's Encrypt. Auto-renew 30d before expiry.", ["devops"]),
    ("do-08", "Backup: RDS automated + manual snapshot before migration.", ["devops"]),
    ("do-09", "Incident: Runbook in docs/incidents/. Post-mortem <48h for P1.", ["devops"]),
    ("do-10", "Cost: Reserved Instances baseline. Spot for dev/staging.", ["devops"]),
    ("te-01", "Backend tests: pytest + pytest-asyncio. Mock external APIs.", ["testing"]),
    ("te-02", "E2E tests: Playwright. Test user flows, not implementation.", ["testing"]),
    ("te-03", "Performance: k6 load tests. Target p95<200ms API.", ["testing"]),
    ("te-04", "Coverage: 80% line coverage gate. SonarQube enforces in CI.", ["testing"]),
    ("te-05", "Contract tests: Pact for provider/consumer testing.", ["testing"]),
]

TEST_QUERIES = [
    ("Create a login form with email validation", ["frontend"],
     ["fe-02", "fe-06", "fe-04"]),
    ("Implement JWT token refresh in HTTP client", ["frontend"],
     ["fe-06", "be-02", "fe-01"]),
    ("Add rate limiting middleware to FastAPI", ["backend"],
     ["be-03", "be-01", "be-04"]),
    ("Set up async background task queue", ["backend"],
     ["be-07", "be-08", "be-01"]),
    ("Write a database migration for user avatar", ["database"],
     ["db-02", "db-03", "db-01"]),
    ("Add full-text search to product catalog", ["database"],
     ["db-10", "db-05", "db-01"]),
    ("Configure GitHub Actions CI pipeline", ["devops"],
     ["do-01", "do-02", "do-03"]),
    ("Set up Kubernetes HPA autoscaling", ["devops"],
     ["do-04", "do-05", "do-01"]),
    ("Write E2E tests for registration flow", ["testing"],
     ["te-02", "te-01"]),
    ("Set up code coverage gate in CI", ["testing"],
     ["te-04", "do-01", "te-01"]),
    ("Add health check endpoint with DB+Redis status", ["backend"],
     ["be-15", "do-05", "db-01"]),
    ("Implement soft delete for user accounts", ["database"],
     ["db-08", "be-06"]),
]


def build_memory():
    mf = MemoryFile(scope=MemoryScope.PROJECT, max_entries=500, max_size_bytes=200*1024)
    for eid, content, domains in ALL_MEMORIES:
        mf.add_entry(MemoryEntry(
            id=eid, scope=MemoryScope.PROJECT,
            category="pattern", content=content, domains=domains,
        ))
    return mf


def evaluate(mf, queries, use_pid: bool, context_usage: float):
    """Run evaluation with given PID setting and context pressure."""
    import tempfile
    from minicode.memory import MemoryManager, MemoryScope

    with tempfile.TemporaryDirectory() as tmp:
        mgr = MemoryManager(project_root=tmp)
        for scope in MemoryScope:
            mgr.memories[scope].entries.clear()
        mgr.memories[MemoryScope.PROJECT] = mf

        ctrl = MemoryInjectionController() if use_pid else _NoopController()
        injector = MemoryInjector(memory_manager=mgr, controller=ctrl, injection_cooldown=0)

        results = {"precision": [], "recall": [], "noise_rate": [], "inject_count": []}

        for query, domains, gt in queries:
            signal = MemoryInjectionSignal(context_usage=context_usage)
            injected = injector.inject_for_task(query, current_files=domains, signal=signal)
            matched_ids = []
            for mem in (injected or []):
                for entry in mf.entries:
                    if entry.content[:80] in mem.content or mem.content[:80] in entry.content:
                        matched_ids.append(entry.id)
                        break

            ids = matched_ids
            hits = sum(1 for eid in ids[:5] if eid in gt)
            results["precision"].append(hits / max(len(ids[:5]), 1) if ids[:5] else 0.0)
            results["recall"].append(hits / len(gt) if gt else 0.0)
            cross = 0
            for eid in ids[:5]:
                entry = next((e for e in mf.entries if e.id == eid), None)
                if entry and not (set(entry.domains) & set(domains)):
                    cross += 1
            results["noise_rate"].append(cross / max(len(ids[:5]), 1) if ids[:5] else 0.0)
            results["inject_count"].append(len(ids))

        return {k: sum(v)/len(v) for k, v in results.items()}


class _NoopController:
    """Static controller: always STANDARD mode, 5 memories."""
    def decide(self, signal, **kwargs):
        from minicode.memory_injector import MemoryInjectionDecision, MemoryInjectionMode
        return MemoryInjectionDecision(
            mode=MemoryInjectionMode.STANDARD,
            max_memories=5, min_relevance=0.3, max_tokens_per_memory=200,
            reasons=["noop"],
        )


class _FakeMemory:
    def __init__(self, mf):
        self._mf = mf
    def search(self, query, scope=None, limit=20, min_relevance=0.1, active_domains=None):
        return self._mf.search(query, active_domains=active_domains)[:limit]
    def search_by_tag(self, scope, tag):
        return [e for e in self._mf.entries if tag in e.tags][:5]
    def get_categories(self, scope):
        return set()
    @property
    def memories(self):
        from minicode.memory import MemoryScope
        class FakeFiles:
            pass
        r = FakeFiles()
        r.__dict__ = {MemoryScope.PROJECT: self._mf}
        return r


def _wrap_memory(mf):
    return _FakeMemory(mf)


def main():
    mf = build_memory()
    print(f"PID Ablation Experiment: {len(TEST_QUERIES)} queries x {len(mf.entries)} memories")
    print()

    # Context pressure levels to test
    levels = [0.3, 0.5, 0.7, 0.9]

    print(f"{'Pressure':>8} | {'PID':^30} | {'No PID':^30}")
    print(f"{'':>8} | {'P@3':>5} {'Noise':>5} {'Inj':>4} | {'P@3':>5} {'Noise':>5} {'Inj':>4}")
    print("-" * 65)

    for cu in levels:
        pid_result = evaluate(mf, TEST_QUERIES, use_pid=True, context_usage=cu)
        nopid_result = evaluate(mf, TEST_QUERIES, use_pid=False, context_usage=cu)

        print(f"{cu*100:>7.0f}% | "
              f"{pid_result['precision']:>5.3f} {pid_result['noise_rate']:>5.0%} {pid_result['inject_count']:>4.1f} | "
              f"{nopid_result['precision']:>5.3f} {nopid_result['noise_rate']:>5.0%} {nopid_result['inject_count']:>4.1f}")

    # Summary
    avg_pid = sum(evaluate(mf, TEST_QUERIES, True, c)['precision'] for c in [0.3,0.5,0.7])/3
    avg_nopid = sum(evaluate(mf, TEST_QUERIES, False, c)['precision'] for c in [0.3,0.5,0.7])/3
    avg_noise_pid = sum(evaluate(mf, TEST_QUERIES, True, c)['noise_rate'] for c in [0.3,0.5,0.7])/3
    avg_noise_nopid = sum(evaluate(mf, TEST_QUERIES, False, c)['noise_rate'] for c in [0.3,0.5,0.7])/3

    print()
    print(f"Average (normal pressure): PID P@3={avg_pid:.3f} vs NoPID P@3={avg_nopid:.3f}")
    print(f"Average noise:            PID {avg_noise_pid:.0%} vs NoPID {avg_noise_nopid:.0%}")
    print()

    # LaTeX table
    print("% --- LaTeX Table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{PID vs static injection across context pressure levels.}")
    print(r"\label{tab:pid_ablation}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Context & \multicolumn{2}{c}{PID} & \multicolumn{2}{c}{No PID} \\")
    print(r"Pressure & P@3 & Noise & P@3 & Noise \\")
    print(r"\midrule")
    for cu in levels:
        pid_r = evaluate(mf, TEST_QUERIES, True, cu)
        nopid_r = evaluate(mf, TEST_QUERIES, False, cu)
        print(f"{cu*100:.0f}\\% & {pid_r['precision']:.3f} & {pid_r['noise_rate']:.1f}\\% & {nopid_r['precision']:.3f} & {nopid_r['noise_rate']:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
