import os

MAIN_BRANCH = "main"


def branch_name(rev="HEAD"):
    return os.popen(f"git rev-parse --abbrev-ref {rev}").read().rstrip()


def common_ancestor_hash(rev1="HEAD", rev2=MAIN_BRANCH):
    if branch_name(rev1) == branch_name(rev2):
        return commit_hash(rev=rev1)

    p = os.popen(
        "diff -u "
        f"<(git rev-list --first-parent {rev1}) "
        f"<(git rev-list --first-parent {rev2}) | "
        "sed -ne 's/^ //p' | head -1"
    )
    return p.read().rstrip()


def commit_hash(rev="HEAD", short=False):
    options = "--short" if short else ""
    return os.popen(f"git rev-parse {options} {rev}").read().rstrip()


def diff(rev="HEAD"):
    return os.popen(f"git --no-pager diff --no-color {rev}").read().rstrip()
