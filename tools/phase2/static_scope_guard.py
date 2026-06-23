# tools/phase2/static_scope_guard.py

import ast
import pathlib
from dataclasses import dataclass
from typing import Tuple, Any

# Public constants
FORBIDDEN_THIRD_PARTY_IMPORT_ROOTS = (
    "torch",
    "numpy",
    "pandas",
    "yaml",
    "sklearn",
    "scipy",
)

FORBIDDEN_PHASE2_RUNTIME_CALLS = (
    "simulate_time_series",
    "build_dataset_in_memory",
    "write_dataset_artifacts",
    "run_phase2_artifact_generation",
)

FORBIDDEN_ARTIFACT_PATH_TOKENS = (
    "phase2_artifacts",
    "P14",
    "P16",
)


# Public dataclasses
@dataclass(frozen=True)
class StaticScopeCheckResult:
    path: str
    forbidden_import_hits: Tuple[str, ...]
    forbidden_call_hits: Tuple[str, ...]
    forbidden_path_token_hits: Tuple[str, ...]
    passed: bool
    reason: str


# Helper Visitor
class ScopeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.import_roots = set()
        self.call_names = set()
        
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            root = name.split('.')[0]
            self.import_roots.add(root)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split('.')[0]
            self.import_roots.add(root)
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            self.call_names.add(func.id)
        elif isinstance(func, ast.Attribute):
            self.call_names.add(func.attr)
        self.generic_visit(node)


# Public functions
def read_text_file(path: str) -> str:
    if type(path) is not str or len(path) == 0:
        raise TypeError("path must be a non-empty str")
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8")


def parse_python_source(source: str) -> ast.AST:
    if type(source) is not str:
        raise TypeError("source must be exactly a str")
    try:
        return ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")


def collect_import_roots(source: str) -> Tuple[str, ...]:
    if type(source) is not str:
        raise TypeError("source must be exactly a str")
    tree = parse_python_source(source)
    visitor = ScopeVisitor()
    visitor.visit(tree)
    return tuple(sorted(list(visitor.import_roots)))


def collect_call_names(source: str) -> Tuple[str, ...]:
    if type(source) is not str:
        raise TypeError("source must be exactly a str")
    tree = parse_python_source(source)
    visitor = ScopeVisitor()
    visitor.visit(tree)
    return tuple(sorted(list(visitor.call_names)))


def collect_string_token_hits(source: str, tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    if type(source) is not str:
        raise TypeError("source must be exactly a str")
    if type(tokens) is not tuple:
        raise TypeError("tokens must be exactly a tuple")
    hits = []
    for tok in tokens:
        if type(tok) is not str or len(tok) == 0:
            raise TypeError("tokens must contain non-empty strings")
        if tok in source:
            hits.append(tok)
    return tuple(sorted(list(set(hits))))


def check_static_scope(
    path: str,
    forbidden_import_roots: Tuple[str, ...] = FORBIDDEN_THIRD_PARTY_IMPORT_ROOTS,
    forbidden_call_names: Tuple[str, ...] = FORBIDDEN_PHASE2_RUNTIME_CALLS,
    forbidden_path_tokens: Tuple[str, ...] = FORBIDDEN_ARTIFACT_PATH_TOKENS,
) -> StaticScopeCheckResult:
    if type(path) is not str or len(path) == 0:
        raise TypeError("path must be a non-empty str")
    if type(forbidden_import_roots) is not tuple:
        raise TypeError("forbidden_import_roots must be a tuple")
    if type(forbidden_call_names) is not tuple:
        raise TypeError("forbidden_call_names must be a tuple")
    if type(forbidden_path_tokens) is not tuple:
        raise TypeError("forbidden_path_tokens must be a tuple")
        
    source = read_text_file(path)
    
    roots = collect_import_roots(source)
    calls = collect_call_names(source)
    tokens = collect_string_token_hits(source, forbidden_path_tokens)
    
    import_hits = tuple(sorted(list(set(roots) & set(forbidden_import_roots))))
    call_hits = tuple(sorted(list(set(calls) & set(forbidden_call_names))))
    token_hits = tokens
    
    passed = (len(import_hits) == 0 and len(call_hits) == 0 and len(token_hits) == 0)
    
    reason = "All static scope checks passed successfully"
    if not passed:
        reason = f"Static scope check failed. Imports: {import_hits}, Calls: {call_hits}, Tokens: {token_hits}"
        
    return StaticScopeCheckResult(
        path=path,
        forbidden_import_hits=import_hits,
        forbidden_call_hits=call_hits,
        forbidden_path_token_hits=token_hits,
        passed=passed,
        reason=reason
    )


def require_static_scope_pass(result: StaticScopeCheckResult) -> None:
    if not isinstance(result, StaticScopeCheckResult):
        raise TypeError("result must be exactly a StaticScopeCheckResult instance")
    if not result.passed:
        raise ValueError(result.reason)
