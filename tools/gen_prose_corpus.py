import json, random
random.seed(7)

lines = []
def emit(text, modules, gold, tag):
    lines.append(json.dumps({"text": text, "modules": modules, "gold": gold, "tag": tag}, ensure_ascii=False))

def edge(a,b): return {"kind":"forbid_edge","from":a,"to":b}
def reach(a,b): return {"kind":"forbid_reach","from":a,"to":b}
def layer(m,al): return {"kind":"layer","module":m,"allowed":al}
def sym(s,ex): return {"kind":"forbid_symbol","symbol":s,"except":ex}

def C(s): lines.append("// "+s)
def BLANK(): lines.append("")

# Operand spellings (varied) for backticked cases (case preserved literally).
names = ["api","db","web","storage","domain","infra","core","plugins","ui","database",
         "handlers","models","service","repository","controllers","parser","renderer",
         "auth","billing","payments","scheduler","worker","presentation","persistence",
         "cli","config","utils","dto","gateway","adapter","ports","usecase","entities",
         "transport","crypto","session","metrics","tracing","cache","queue",
         "data_access","view-model","app.core","mod1","pkg_b","x","y","http_client","grpc"]

pairs = []
random.shuffle(names)
for i in range(0, len(names)-1, 2):
    pairs.append((names[i], names[i+1]))

# ---------------- backticked direct edges (forbid_edge_re) ----------------
C("=== GENERATED: backticked direct edges (forbid_edge_re) ===")
edge_modals = ["must not","should not","may not","can not","cannot","can't","does not","do not"]
edge_verbs  = ["import","imports","depend on","depends on","access","accesses","touch","touches"]
nouns = ["", " layer", " module", " package", " crate", " component"]
for idx,(a,b) in enumerate(pairs):
    m = edge_modals[idx % len(edge_modals)]
    v = edge_verbs[idx % len(edge_verbs)]
    n = nouns[idx % len(nouns)]
    emit(f"`{a}`{n} {m} {v} `{b}`.", [a,b], [edge(a,b)], f"gen-edge-{idx}")
BLANK()

# ---------------- backticked "never" edges (never_edge_re) ----------------
C("=== GENERATED: backticked never edges (never_edge_re) ===")
never_verbs = ["imports","depends on","uses","references"]
for idx,(a,b) in enumerate(pairs[:8]):
    v = never_verbs[idx % len(never_verbs)]
    emit(f"`{a}` never {v} `{b}`.", [a,b], [edge(a,b)], f"gen-never-{idx}")
BLANK()

# ---------------- backticked transitive reach (forbid_reach_re) ----------------
C("=== GENERATED: backticked transitive reach (forbid_reach_re) ===")
reach_modals = ["must not","should not","may not"]
reach_mid = ["transitively import","transitively depend on","indirectly use",
             "indirectly reference","even transitively reach","reach"]
for idx,(a,b) in enumerate(pairs[:12]):
    m = reach_modals[idx % len(reach_modals)]
    mid = reach_mid[idx % len(reach_mid)]
    emit(f"`{a}` {m} {mid} `{b}`.", [a,b], [reach(a,b)], f"gen-reach-{idx}")
BLANK()

# ---------------- backticked reverse "imported by" (forbid_by_re) ----------------
C("=== GENERATED: backticked reverse imported-by edges (forbid_by_re) ===")
by_modals = ["must not","should not","may not","can not"]
by_verbs  = ["imported","used","referenced","accessed","depended on"]
by_preps  = ["by","from","in"]
for idx,(a,b) in enumerate(pairs[:10]):
    m = by_modals[idx % len(by_modals)]
    v = by_verbs[idx % len(by_verbs)]
    p = by_preps[idx % len(by_preps)]
    # `a` must not be <v> <p> `b`  =>  edge b->a
    emit(f"`{a}` {m} be {v} {p} `{b}`.", [a,b], [edge(b,a)], f"gen-by-{idx}")
BLANK()

# ---------------- backticked symmetric independence (independent_re) ----------------
C("=== GENERATED: backticked symmetric independence (independent_re; two edges) ===")
indep_verbs = ["is independent of","are independent from","stays independent of","remains independent from"]
for idx,(a,b) in enumerate(pairs[:8]):
    v = indep_verbs[idx % len(indep_verbs)]
    emit(f"`{a}` {v} `{b}`.", [a,b], [edge(a,b), edge(b,a)], f"gen-indep-{idx}")
BLANK()

# ---------------- backticked depends-on-nothing (depends_nothing_re) ----------------
C("=== GENERATED: backticked depends-on-nothing layers (depends_nothing_re) ===")
nothing_tails = ["depends on nothing","imports nothing","has no dependencies","has no imports","has no deps"]
for idx,a in enumerate([p[0] for p in pairs[:10]]):
    t = nothing_tails[idx % len(nothing_tails)]
    emit(f"`{a}` {t}.", [a], [layer(a,[])], f"gen-nothing-{idx}")
BLANK()

# ---------------- backticked "only depends on" (only_depends_re) ----------------
C("=== GENERATED: backticked only-depends layers (only_depends_re) ===")
only_modals = ["may only","must only","can only","should only","only"]
only_verbs  = ["depend on","import","depends on","imports"]
for idx,(a,b) in enumerate(pairs[:8]):
    m = only_modals[idx % len(only_modals)]
    v = only_verbs[idx % len(only_verbs)]
    emit(f"`{a}` {m} {v} `{b}`.", [a,b], [layer(a,[b])], f"gen-only-{idx}")
# multi-target
emit("`app` may only depend on `core`, `utils`, and `config`.", ["app","core","utils","config"], [layer("app",["core","utils","config"])], "gen-only-multi")
emit("`web` should only import `api` and `dto`.", ["web","api","dto"], [layer("web",["api","dto"])], "gen-only-multi2")
BLANK()

# ---------------- backticked forbidden symbols (forbid_symbol_re) ----------------
C("=== GENERATED: backticked forbidden symbols (forbid_symbol_re) ===")
symbols = ["eval","exec","os.environ","System.exit","println","unwrap","panic!","global_state","time.now","math.random"]
sym_lead_plain = ["Must not use","Should not call","May not invoke","Never use","Never call","Don't use","Don't call"]
for idx,s in enumerate(symbols):
    lead = sym_lead_plain[idx % len(sym_lead_plain)]
    emit(f"{lead} `{s}`.", [], [sym(s,[])], f"gen-symbol-{idx}")
# with except/outside
emit("No direct `os.environ` outside `config`.", ["config"], [sym("os.environ",["config"])], "gen-symbol-outside")
emit("No raw `sql` except in `repository`.", ["repository"], [sym("sql",["repository"])], "gen-symbol-except")
emit("No use of `println` outside `logger`.", ["logger"], [sym("println",["logger"])], "gen-symbol-no-use-of")
emit("Modules may not reference `singleton`.", [], [sym("singleton",[])], "gen-symbol-may-not-ref")
BLANK()

# ---------------- BARE edges (extract_bare_rules) ----------------
C("=== GENERATED: bare-operand direct edges (no backticks; grounding-gated) ===")
bare_modals = ["must not","should not","may not","can not","cannot","can't"]
bare_verbs  = ["import","imports","depend on","depends on","reference","references","access","accesses","use","uses"]
bare_nouns  = ["", " layer", " module", " package", " crate", " component", " code", " library"]
random.shuffle(names)
bpairs = []
for i in range(0, len(names)-1, 2):
    bpairs.append((names[i], names[i+1]))
# bare grounds case-insensitively but emits real segment; keep operands lowercase & plain (no dots/dashes that break token regex boundaries are fine, but '.' splits? token is [a-z][\w.-]* so dots/dashes allowed). Avoid 'app.core' as subject start fine.
# Bare operands must ground AND survive the BARE_STOPWORDS denylist, so drop any
# name that collides with a stopword (e.g. "entities") — those are negatives, not
# positives, for the bare extractor.
BARE_STOPWORDS = {"entities","entity","module","modules","layer","layers","package",
 "packages","crate","crates","component","components","code","library","libraries",
 "class","classes","interface","interfaces","data","type","types","system","systems"}
bare_ok_names = [n for n in names
                 if n.replace("_","").replace("-","").replace(".","").isalnum()
                 and n[0].isalpha()
                 and n.lower() not in BARE_STOPWORDS]
bpairs = []
for i in range(0, len(bare_ok_names)-1, 2):
    bpairs.append((bare_ok_names[i], bare_ok_names[i+1]))
for idx,(a,b) in enumerate(bpairs):
    if a==b: continue
    m = bare_modals[idx % len(bare_modals)]
    v = bare_verbs[idx % len(bare_verbs)]
    n = bare_nouns[idx % len(bare_nouns)]
    art = "the " if idx%2 else ""
    emit(f"{art.capitalize() if art else ''}{a}{n} {m} {v} {b}.".strip(), [a,b], [edge(a,b)], f"gen-bare-edge-{idx}")
BLANK()

# bold + path-style bare
C("=== GENERATED: bare edges with bold / path-style operands ===")
emit("The **repository** layer cannot reference **service**.", ["repository","service"], [edge("repository","service")], "gen-bare-bold")
emit("**handlers** must not access **models**.", ["handlers","models"], [edge("handlers","models")], "gen-bare-bold-both")
emit("domain/model must not import infra/db.", ["domain/model","infra/db"], [edge("domain/model","infra/db")], "gen-bare-path")
emit("The Auth module must not depend on Billing.", ["auth","billing"], [edge("auth","billing")], "gen-bare-mixedcase")
emit("src/web should not import src/db.", ["src/web","src/db"], [edge("src/web","src/db")], "gen-bare-path2")
BLANK()

# ---------------- BARE reach ----------------
C("=== GENERATED: bare-operand transitive reach ===")
bare_reach_mid = ["transitively import","transitively depend on","indirectly use","indirectly reference","reach"]
for idx,(a,b) in enumerate(bpairs[:8]):
    if a==b: continue
    m = bare_modals[idx % 3]  # must/should/may not (avoid cannot+reach quirk on bare? bare reach allows cannot but keep simple)
    mid = bare_reach_mid[idx % len(bare_reach_mid)]
    emit(f"{a} {m} {mid} {b}.", [a,b], [reach(a,b)], f"gen-bare-reach-{idx}")
BLANK()

# ---------------- KNOWN MISSES (intended FN) ----------------
C("=== known misses: real rules current extractors cannot catch (intended FN) ===")
emit("Core internals must not depend on satellite implementation classes.", ["core","satellite"], [edge("core","satellite")], "miss-multiword-operand")
emit("`api` must not import anything from `db`.", ["api","db"], [edge("api","db")], "miss-import-anything-from")
emit("Modules in `api` should not call into `db`.", ["api","db"], [edge("api","db")], "miss-call-into")
BLANK()

# ---------------- HARD NEGATIVES (must extract nothing) ----------------
C("=== hard negatives: boilerplate / descriptive prose that must NOT extract ===")
negs = [
 ("High-level modules should not depend on low-level modules.", [], "noise-solid"),
 ("Clients SHOULD NOT depend on the transport details.", ["transport"], "noise-rfc"),
 ("Untrusted code must not access the kernel.", [], "noise-security"),
 ("This module should be easy to test.", ["testing"], "noise-generic"),
 ("Users should not have to read the source.", [], "noise-prose"),
 ("The README should not reference outdated commands.", ["readme","commands"], "noise-ungrounded-operand"),
 ("Components must not depend on implementation details.", [], "noise-stopword-operands"),
 ("Each service should be independently deployable.", ["service"], "noise-no-rule"),
 ("The application should not depend on external state.", [], "noise-ungrounded-target"),
 ("Avoid circular dependencies between packages.", [], "noise-advice"),
 ("It is recommended that `api` not import `db`.", ["api","db"], "noise-no-modal"),
 ("See `api` and `db` for details.", ["api","db"], "noise-stray-backticks"),
 ("`api` imports `db` for serialization.", ["api","db"], "noise-descriptive-import"),
 ("The `db` module is imported by many `api` handlers.", ["db","api"], "noise-descriptive-by"),
 ("`domain` only contains pure functions.", ["domain"], "noise-only-contains"),
 ("`config` is independent and reusable.", ["config"], "noise-independent-no-target"),
 ("Don't forget to update `db`.", ["db"], "noise-dont-forget"),
 ("No more than `3` retries allowed.", [], "noise-no-more-than"),
 ("We must not break `api` compatibility.", ["api"], "noise-must-not-break"),
 ("The frontend talks to the backend via HTTP.", ["frontend","backend"], "noise-bare-no-verb"),
 ("Avoid using `eval` where possible.", [], "noise-avoid-using"),
 ("payments must not import billing.", [], "noise-bare-ungrounded"),
 ("core must not depend on core.", ["core"], "noise-bare-self-edge"),
 ("Services must not depend on each other.", ["services"], "noise-bare-stopword-target"),
 ("The auth module should be well tested.", ["auth"], "noise-bare-no-prohibition"),
 ("billing must not leak payments data.", ["billing","payments"], "noise-bare-non-dep-verb"),
 ("Higher layers must not depend on lower layers.", [], "noise-bare-solid"),
 ("The system must not crash on startup.", [], "noise-bare-non-dep-verb2"),
 ("Functions should not be too long.", [], "noise-generic2"),
 ("Prefer composition over inheritance.", [], "noise-advice2"),
 ("The `cache` should be invalidated on write.", ["cache"], "noise-descriptive-should"),
 ("Nothing should depend on `internal`.", ["internal"], "noise-nothing-subject"),
 ("`api` and `web` both import `shared`.", ["api","web","shared"], "noise-descriptive-multi"),
]
for text, mods, tag in negs:
    emit(text, mods, [], tag)

print("\n".join(lines))
print(f"// total non-comment cases: {sum(1 for l in lines if l.startswith('{'))}", file=__import__('sys').stderr)
