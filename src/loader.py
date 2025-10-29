import pathlib, sys
def resolve_repo():
    repo_root = pathlib.Path.cwd().parents[0]
    sys.path.append(str(repo_root))
    print(repo_root)
