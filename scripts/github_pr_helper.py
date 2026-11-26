#!/usr/bin/env python3
"""Maker/Checker GitHub PR helper and lingering-branch auditor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def audit_prs() -> int:
    """Verify zero open PRs and zero lingering remote feature branches."""
    pr_res = _run(["gh", "pr", "list", "--state", "open", "--json", "number,title,headRefName"])
    if pr_res.returncode != 0:
        sys.stderr.write(f"gh pr list failed: {pr_res.stderr}\n")
        return 1
    open_prs = json.loads(pr_res.stdout or "[]")

    br_res = _run(["git", "branch", "-r", "--format=%(refname:short)"])
    if br_res.returncode != 0:
        sys.stderr.write(f"git branch -r failed: {br_res.stderr}\n")
        return 1
    remote_branches = [
        line.strip()
        for line in (br_res.stdout or "").splitlines()
        if line.strip() and line.strip() not in {"origin/main", "origin/HEAD", "origin"}
    ]

    if open_prs:
        sys.stderr.write(f"Found open PRs: {open_prs}\n")
        return 1
    if remote_branches:
        sys.stderr.write(f"Found lingering remote feature branches: {remote_branches}\n")
        return 1

    print("Audit passed: Zero open pull requests, Zero lingering remote feature branches.")
    return 0


def create_pr(title: str, body: str, base: str = "main") -> int:
    """Create a GitHub PR via gh CLI."""
    res = _run(["gh", "pr", "create", "--base", base, "--title", title, "--body", body])
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    return res.returncode


def merge_pr(pr_number: int, *, delete_branch: bool = True) -> int:
    """Squash-merge a GitHub PR via gh CLI."""
    cmd = ["gh", "pr", "merge", str(pr_number), "--squash"]
    if delete_branch:
        cmd.append("--delete-branch")
    res = _run(cmd)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="BeliefAudit Maker/Checker PR helper")
    parser.add_argument("--audit", action="store_true", help="Audit open PRs and remote branches")
    parser.add_argument("--create", action="store_true", help="Create a PR")
    parser.add_argument("--merge", action="store_true", help="Squash-merge a PR")
    parser.add_argument("--pr", type=int, help="Pull request number for --merge")
    parser.add_argument("--title", default="", help="PR title for --create")
    parser.add_argument("--body", default="", help="PR body for --create")
    parser.add_argument("--base", default="main", help="Base branch for --create")
    parser.add_argument(
        "--delete-branch", action="store_true", help="Delete remote branch on merge"
    )
    args = parser.parse_args()

    if args.audit:
        return audit_prs()
    if args.create:
        if not args.title:
            sys.stderr.write("--title is required with --create\n")
            return 2
        return create_pr(args.title, args.body, base=args.base)
    if args.merge:
        if args.pr is None:
            sys.stderr.write("--pr is required with --merge\n")
            return 2
        return merge_pr(args.pr, delete_branch=args.delete_branch)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
