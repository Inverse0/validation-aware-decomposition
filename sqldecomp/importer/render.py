"""A readable dump of an ImportResult, for inspection and debugging. Reads the model only, so it works on any result."""
from __future__ import annotations

from sqldecomp.model import Filter, ImportResult, Join, Op1, Op2, Plan, Scan, plan_children, region_inputs_under


def regions(result: ImportResult) -> list[frozenset[str]]:
    """The inner-join regions of a result, each as the set of its input ids. A region is a maximal tree of Joins and
    Filters over registered inputs; an input with nothing of the kind above it is a region of size one."""
    by_plan = {i.plan: i.id for i in result.facts.inputs}
    found: list[frozenset[str]] = []

    def visit(p: Plan, in_region: bool) -> None:
        if p in by_plan:  # a leaf of the region above it; whatever is below starts new regions
            if not in_region:
                found.append(frozenset({by_plan[p]}))
            in_region = False
        elif isinstance(p, (Join, Filter)):
            if not in_region:
                ids = region_inputs_under(p, by_plan)
                if ids is not None:
                    found.append(ids)
            in_region = True
        else:
            in_region = False
        for c in plan_children(p):
            visit(c, in_region)

    for s in result.scopes:
        visit(s.plan, False)
    unique: list[frozenset[str]] = []
    for r in found:  # a shared CTE plan is reached once per use
        if r not in unique:
            unique.append(r)
    return unique


def render(result: ImportResult) -> str:
    side, facts = result.side, result.facts
    scope_of_plan = {s.plan: s for s in result.scopes}
    owners = facts.owners
    edges = {e.conjunct: e.inputs for e in facts.edges}
    out: list[str] = ["SQL: " + " ".join(result.sql.split())]

    def conj(cid: str) -> str:
        e = side[cid]
        role = f"join conjunct over {', '.join(edges[cid])}" if cid in edges else \
            f"local, owner {owners[cid]}" if cid in owners else "no owner: stays on top of its region"
        extra = "".join(f"; {k} {e.data[k]}" for k in ("outer_refs", "subqueries") if e.data.get(k))
        return f"{cid}: {e.sql}   [{e.data.get('source')}; {role}{extra}]"

    def nested(plan: Plan) -> str:
        s = scope_of_plan.get(plan)
        return f"-> the plan of scope {s.id}" if s is not None else "-> (a plan that is no scope's plan)"

    def show(p: Plan, depth: int) -> None:
        pad = "  " * depth
        if isinstance(p, Scan):
            out.append(f"{pad}Scan {side[p.occ].sql or p.table}   [{p.occ}]")
        elif isinstance(p, Join):
            out.append(f"{pad}Join" + (" ON" if p.conjuncts else "  (no condition: a cross join the query contains)"))
            for c in p.conjuncts:
                out.append(f"{pad}  . {conj(c)}")
            show(p.left, depth + 1)
            show(p.right, depth + 1)
        elif isinstance(p, Filter):
            out.append(f"{pad}Filter {conj(p.conjunct)}")
            show(p.input, depth + 1)
        elif isinstance(p, Op1):
            e = side[p.op]
            if e.kind == "ref":
                what = f"CTE {e.data['name']} (declared in scope {e.data.get('declared_in')})" if e.data.get("name") else "derived table"
                extra = f"; outer_refs {e.data['outer_refs']}" if e.data.get("outer_refs") else ""
                out.append(f"{pad}Op1 {p.op}: use of {what} as {e.data['alias']}{extra}   {nested(p.input)}")
            else:
                extra = "".join(f"; {k} {e.data[k]}" for k in ("outer_refs", "subqueries") if e.data.get(k))
                if e.data.get("key"):
                    extra += "; unique together: " + ", ".join(".".join(k) for k in e.data["key"])
                out.append(f"{pad}Op1 {p.op}: {e.sql}{extra}")
                show(p.input, depth + 1)
        elif isinstance(p, Op2):
            e = side[p.op]
            if e.kind == "setop":
                out.append(f"{pad}Op2 {p.op}: {e.sql}")
                out.append(f"{pad}  left  {nested(p.left)}")
                out.append(f"{pad}  right {nested(p.right)}")
            else:
                extra = "".join(f"; {k} {e.data[k]}" for k in ("refs", "exposes", "outer_refs", "subqueries") if e.data.get(k))
                out.append(f"{pad}Op2 {p.op}: {e.sql}{extra}")
                show(p.left, depth + 1)
                show(p.right, depth + 1)

    for s in result.scopes:
        head = f"scope {s.id} ({s.kind}" + (f" {s.name}" if s.name else "") + (f", inside scope {s.parent}" if s.parent else "") + ")"
        out += ["", head, "  outputs: " + ", ".join(f"{o.name} = {o.sql}" for o in s.outputs), "  plan:"]
        show(s.plan, 2)

    order = {i.id: n for n, i in enumerate(facts.inputs)}
    out += ["", "regions (inputs of each): " + "  ".join("{" + ", ".join(sorted(r, key=order.get)) + "}" for r in regions(result))]
    uses: dict[str, list[str]] = {}
    for e in side.values():
        if e.kind == "ref":
            uses.setdefault(e.data["scope"], []).append(f"{e.id} (as {e.data['alias']})")
    for sid, refs in uses.items():
        if len(refs) > 1:
            out.append(f"shared: the ONE plan of scope {sid} is used by {len(refs)} refs: " + ", ".join(refs))
    if facts.equates:
        out.append("equates: " + "; ".join(f"{q.conjunct}: {q.left.origin}.{q.left.column} = {q.right.origin}.{q.right.column}"
                                           for q in facts.equates))
    for op, cols in facts.group_cols.items():
        out.append(f"group_cols {op}: " + (", ".join(f"{c.origin}.{c.column}" for c in cols) or "(none: one group)"))
    if facts.keeps:
        out.append("keeps: " + "; ".join(f"{k.op}.{k.name} <- {k.source.origin}.{k.source.column}" for k in facts.keeps))
    return "\n".join(out)
