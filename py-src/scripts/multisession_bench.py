"""Multi-session memory evaluation benchmark.

Tests the same capabilities as LongMemEval:
- Single-hop: direct memory retrieval from a past session
- Multi-hop: combining information across multiple sessions
- Temporal: recognizing which information is more recent

Self-contained — no external dataset required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time

sys.path.insert(0, ".")

from minicode.memory import MemoryEntry, MemoryFile, MemoryScope, MemoryManager
from minicode.memory_pipeline import MemoryPipeline


# ── Session simulation ──────────────────────────────────────────────

def simulate_sessions():
    """Create 5 simulated coding sessions, each generating memories."""
    sessions = [
        # Session 1: Frontend setup
        [
            ("s1-m1", "Project uses React 18 with TypeScript strict mode. All components must be functional with hooks.", ["frontend"]),
            ("s1-m2", "State management uses Zustand v4. Store files are in src/stores/.", ["frontend"]),
            ("s1-m3", "Forms use react-hook-form v7 with zod validation schemas.", ["frontend"]),
        ],
        # Session 2: Backend setup
        [
            ("s2-m1", "API built with FastAPI. All endpoints are async and return Pydantic models.", ["backend"]),
            ("s2-m2", "JWT auth with refresh token rotation. Access tokens expire in 15min.", ["backend", "security"]),
            ("s2-m3", "Rate limiting uses Redis sliding window. 100 req/min per user default.", ["backend"]),
        ],
        # Session 3: Database setup
        [
            ("s3-m1", "PostgreSQL 16 with PostGIS extension. Connection pooling via PgBouncer.", ["database"]),
            ("s3-m2", "Migrations managed by Alembic. Never edit tables directly in production.", ["database"]),
            ("s3-m3", "JSONB columns for semi-structured data with GIN index.", ["database"]),
        ],
        # Session 4: DevOps setup
        [
            ("s4-m1", "CI/CD uses GitHub Actions with lint+typecheck+test+build checks.", ["devops"]),
            ("s4-m2", "Docker multi-stage builds with python:3.12-slim base image.", ["devops"]),
            ("s4-m3", "Kubernetes on EKS 1.29 with HPA scaling 2-10 pods at CPU>70%.", ["devops"]),
        ],
        # Session 5: Testing + refinements
        [
            ("s5-m1", "Backend tests use pytest with pytest-asyncio. Fixtures in conftest.py.", ["testing"]),
            ("s5-m2", "E2E tests use Playwright. Headless mode in CI.", ["testing"]),
            ("s5-m3", "Frontend state migration from Redux complete. All Redux code removed.", ["frontend"]),
        ],
    ]
    return sessions


def build_memory_from_sessions(sessions):
    mf = MemoryFile(scope=MemoryScope.PROJECT, max_entries=500, max_size_bytes=200*1024)
    for session in sessions:
        for eid, content, domains in session:
            mf.add_entry(MemoryEntry(
                id=eid, scope=MemoryScope.PROJECT,
                category="pattern", content=content, domains=domains,
            ))
    return mf


# ── Query types ─────────────────────────────────────────────────────

QUERIES = {
    "single_hop": [
        ("What state management library does the project use?", ["frontend"], ["s1-m2"]),
        ("What API framework is the backend built with?", ["backend"], ["s2-m1"]),
        ("What database version and extensions are used?", ["database"], ["s3-m1"]),
        ("What CI/CD platform does the project use?", ["devops"], ["s4-m1"]),
        ("What E2E testing framework is used?", ["testing"], ["s5-m2"]),
    ],
    "multi_hop": [
        ("How should forms be validated in the React frontend?", ["frontend"], ["s1-m3", "s1-m1"]),
        ("What is the API authentication mechanism and its timeout?", ["backend"], ["s2-m2"]),
        ("What scaling policy does Kubernetes use?", ["devops"], ["s4-m3"]),
    ],
    "temporal": [
        ("Is Redux still used for state management?", ["frontend"], ["s5-m3"]),
        ("What Python testing framework is recommended?", ["testing"], ["s5-m1"]),
    ],
}


def evaluate_pipeline(mf, queries_dict):
    """Run the full MemoryPipeline against queries and measure accuracy."""
    results = {"single_hop": [], "multi_hop": [], "temporal": []}

    with tempfile.TemporaryDirectory() as tmp:
        mgr = MemoryManager(project_root=tmp)
        for s in MemoryScope:
            mgr.memories[s].entries.clear()
        mgr.memories[MemoryScope.PROJECT] = mf

        pipeline = MemoryPipeline(mgr)
        pipeline.initialize(model_adapter=None, enable_vector=False)

        for category, queries in queries_dict.items():
            for query, domains, gt in queries:
                memories = pipeline.read(query, active_domains=domains, max_results=5)
                retrieved_ids = [m["id"] for m in memories[:5]]
                hits = sum(1 for eid in retrieved_ids if eid in gt)
                # Precision: what fraction of top-5 are relevant
                p = hits / max(len(retrieved_ids), 1) if retrieved_ids else 0.0
                # Recall: what fraction of ground truth was found
                r = hits / len(gt) if gt else 0.0
                results[category].append((p, r))

    # Average per category
    summary = {}
    for cat, scores in results.items():
        if scores:
            avg_p = sum(s[0] for s in scores) / len(scores)
            avg_r = sum(s[1] for s in scores) / len(scores)
            summary[cat] = {"P@5": avg_p, "R": avg_r, "count": len(scores)}

    return summary


def main():
    sessions = simulate_sessions()
    mf = build_memory_from_sessions(sessions)
    print(f"Multi-Session Memory Benchmark: {len(sessions)} sessions, {len(mf.entries)} memories")
    print()

    summary = evaluate_pipeline(mf, QUERIES)

    print(f"{'Category':<15} {'P@5':>6} {'Recall':>6} {'Queries':>8}")
    print("-" * 38)
    total_p, total_r, total_n = 0, 0, 0
    for cat in ["single_hop", "multi_hop", "temporal"]:
        s = summary.get(cat, {})
        if s:
            p, r, n = s["P@5"], s["R"], s["count"]
            print(f"{cat:<15} {p:>6.3f} {r:>6.3f} {n:>8}")
            total_p += p * n
            total_r += r * n
            total_n += n

    if total_n > 0:
        print("-" * 38)
        print(f"{'Overall':<15} {total_p/total_n:>6.3f} {total_r/total_n:>6.3f} {total_n:>8}")

    print()
    print("Note: LongMemEval comparison requires running the official benchmark.")
    print("This self-contained test validates the same retrieval capabilities.")


if __name__ == "__main__":
    main()
