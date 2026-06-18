use super::*;
use crate::code::symbol::DepEdge;
use crate::findings::Verdict;

fn dep(from: &str, to: &str) -> DepEdge {
    DepEdge {
        from_module: from.to_string(),
        to_module: to.to_string(),
    }
}

/// Build a synthetic index: `real` are the actual module edges; every name in
/// `modules` is forced into `module_set()` so it grounds (mirrors the helper in
/// `rules::tests`).
fn idx(real: &[(&str, &str)], modules: &[&str]) -> CodeIndex {
    CodeIndex {
        symbols: vec![],
        edges: modules.iter().map(|m| dep(m, "_")).collect(),
        module_edges: real.iter().map(|(a, b)| dep(a, b)).collect(),
        ref_edges: vec![],
    }
}

fn mermaid(body: &str) -> String {
    format!("```mermaid\n{body}\n```\n")
}

// ---- the set-diff ---------------------------------------------------------

#[test]
fn phantom_edge_is_contradicted() {
    let index = idx(
        &[("src/api", "src/domain")],
        &["src/api", "src/domain", "src/db"],
    );
    let md = mermaid("graph TD\n  api[src/api] --> db[src/db]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].verdict, Verdict::Contradicted);
    assert_eq!(f[0].code_refs, vec!["src/api -> src/db"]);
}

#[test]
fn drawn_real_edge_is_clean() {
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    let md = mermaid("graph TD\n  api[src/api] --> domain[src/domain]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
}

#[test]
fn ungrounded_endpoint_is_skipped() {
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    // `User` matches no module → the edge is external, not a phantom; and `User`
    // has no path separator so it is not a stale box either.
    let md = mermaid("graph TD\n  User --> api[src/api]");
    assert!(
        check(&md, "doc.md", &index, std::path::Path::new(".")).is_empty(),
        "{:?}",
        check(&md, "doc.md", &index, std::path::Path::new("."))
    );
}

#[test]
fn undirected_edge_matches_either_direction() {
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    // Drawn undirected; real import runs api->domain. Should be clean.
    let md = mermaid("graph TD\n  domain[src/domain] --- api[src/api]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
}

#[test]
fn stale_box_with_path_label_is_stale() {
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    // `src/legacy` resolves to no module; the `/` marks it as module-intent.
    let md = mermaid("graph TD\n  api[src/api] --> legacy[src/legacy]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    // api->legacy is skipped (legacy ungrounded); the box itself is stale.
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].verdict, Verdict::Stale);
    assert!(f[0].claim.contains("src/legacy"));
}

#[test]
fn missing_arrow_between_drawn_boxes_is_undocumented() {
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    // Both boxes drawn, but the real api->domain import is not drawn.
    let md = mermaid("graph TD\n  api[src/api]\n  domain[src/domain]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert_eq!(f.len(), 1, "{f:?}");
    assert_eq!(f[0].verdict, Verdict::Undocumented);
    assert_eq!(f[0].code_refs, vec!["src/api -> src/domain"]);
}

#[test]
fn omitted_module_does_not_trigger_missing_arrow() {
    // domain isn't in the diagram at all → no missing-arrow noise.
    let index = idx(&[("src/api", "src/domain")], &["src/api", "src/domain"]);
    let md = mermaid("graph TD\n  api[src/api] --> other\n");
    // api->other is phantom only if `other` grounds; it doesn't, so skipped.
    assert!(check(&md, "doc.md", &index, std::path::Path::new(".")).is_empty());
}

// ---- label grounding: regressions from the 10-repo wild audit -------------

#[test]
fn box_label_with_source_extension_grounds() {
    // Wild FP (novu): `pipeline/runner.ts` is drawn but the module is stored
    // extension-stripped as `pipeline/runner`, so the box must NOT read stale.
    let index = idx(&[], &["src/commands/wizard/pipeline/runner"]);
    let md = mermaid("graph TD\n  d[pipeline/runner.ts]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
}

#[test]
fn url_route_box_is_not_a_stale_module() {
    // Wild FP (fastapi): REST route boxes in a request-flow diagram.
    let index = idx(&[], &["src/api"]);
    let md = mermaid("graph TD\n  a[/items/public/] --> b[/users/{user_id}]");
    assert!(
        check(&md, "doc.md", &index, std::path::Path::new(".")).is_empty(),
        "{:?}",
        check(&md, "doc.md", &index, std::path::Path::new("."))
    );
}

#[test]
fn decision_node_label_is_not_a_stale_module() {
    // Wild FP (airflow): Mermaid decision-node label with `<br/>` and `=`.
    let index = idx(&[], &["src/api"]);
    let md = mermaid("graph TD\n  a[full_tests_needed=False<br/>is_canary_run=True]");
    assert!(
        check(&md, "doc.md", &index, std::path::Path::new(".")).is_empty(),
        "{:?}",
        check(&md, "doc.md", &index, std::path::Path::new("."))
    );
}

#[test]
fn phantom_edge_grounds_through_extension() {
    // Both endpoints drawn with `.ts`; the import is real → clean, not phantom.
    let index = idx(&[("src/a", "src/b")], &["src/a", "src/b"]);
    let md = mermaid("graph TD\n  x[src/a.ts] --> y[src/b.ts]");
    let f = check(&md, "doc.md", &index, std::path::Path::new("."));
    assert!(f.iter().all(|x| !x.verdict.is_reportable()), "{f:?}");
}

// ---- parsers --------------------------------------------------------------

#[test]
fn mermaid_parses_nodes_and_edges() {
    let d = mermaid::parse("graph LR\n  a[Auth] --> b[DB]\n  b --> c", "o").unwrap();
    assert_eq!(d.kind, DiagramKind::Flowchart);
    assert_eq!(d.edges.len(), 2);
    assert!(d.edges.iter().all(|e| e.directed));
    assert_eq!(d.text("a"), "Auth");
}

#[test]
fn mermaid_skips_sequence_diagram() {
    assert!(mermaid::parse("sequenceDiagram\n  A->>B: hi", "o").is_none());
}

#[test]
fn mermaid_undirected_edge() {
    let d = mermaid::parse("graph TD\n  a --- b", "o").unwrap();
    assert_eq!(d.edges.len(), 1);
    assert!(!d.edges[0].directed);
}

#[test]
fn plantuml_parses_components() {
    let d = plantuml::parse("@startuml\n[Auth] --> [DB]\n@enduml", "o").unwrap();
    assert_eq!(d.edges.len(), 1);
    assert_eq!(d.text(&d.edges[0].from), "Auth");
}

#[test]
fn plantuml_skips_sequence() {
    assert!(plantuml::parse("@startuml\nAlice -> Bob : hello\n@enduml", "o").is_none());
}

#[test]
fn dot_digraph_is_directed() {
    let d = dot::parse("digraph G {\n  \"a\" -> \"b\";\n}", "o").unwrap();
    assert_eq!(d.edges.len(), 1);
    assert!(d.edges[0].directed);
    assert_eq!(d.edges[0].from, "a");
}

#[test]
fn dot_graph_is_undirected() {
    let d = dot::parse("graph G {\n  a -- b;\n}", "o").unwrap();
    assert_eq!(d.edges.len(), 1);
    assert!(!d.edges[0].directed);
}

#[test]
fn dot_node_label() {
    let d = dot::parse("digraph { a [label=\"Auth\"]; a -> b; }", "o").unwrap();
    assert_eq!(d.text("a"), "Auth");
}

// ---- source extraction ----------------------------------------------------

#[test]
fn sources_extracts_each_format() {
    let md = "intro\n```mermaid\ngraph TD\n a-->b\n```\ntext\n```dot\ndigraph{a->b}\n```\n@startuml\n[A]-->[B]\n@enduml\n";
    let got = sources(md);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0].format, Format::Mermaid);
    assert_eq!(got[1].format, Format::Dot);
    assert_eq!(got[2].format, Format::PlantUml);
}

#[test]
fn sources_ignores_non_diagram_fence() {
    let md = "```rust\n@startuml not a diagram\nlet x = 1;\n```\n";
    assert!(sources(md).is_empty());
}

/// End-to-end dogfood: build a *real* index via tree-sitter from temp source,
/// then run class + sequence diagrams through the full `check` dispatch. Proves
/// `Symbol.calls`/`members` populate and grounding works on real extraction
/// (the per-module tests use synthetic indexes).
#[test]
fn class_and_sequence_ground_against_a_real_index() {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("staleguard-dogfood-{nanos}"));
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("src/svc.rs"),
        "pub struct Service {}\n\
         impl Service {\n\
         \x20   pub fn validate(&self) {}\n\
         \x20   pub fn save(&self) {}\n\
         }\n\
         pub fn handle() {\n\
         \x20   step_one();\n\
         \x20   step_two();\n\
         }\n\
         fn step_one() {}\n\
         fn step_two() {}\n",
    )
    .unwrap();

    let index = crate::code::CodeIndex::build(&dir);

    // Class diagram: `validate` is real, `purge` is not.
    let class = mermaid("classDiagram\n  class Service {\n    +validate()\n    +purge()\n  }");
    let cf = check(&class, "doc.md", &index, &dir);
    assert!(
        cf.iter()
            .any(|f| f.verdict == Verdict::Supported && f.claim.contains("validate")),
        "{cf:?}"
    );
    assert!(
        cf.iter()
            .any(|f| f.verdict == Verdict::Stale && f.detail.contains("purge")),
        "{cf:?}"
    );

    // Sequence diagram: `handle` calls step_one then step_two, in order.
    let seq = mermaid("sequenceDiagram\n  H->>S: step_one()\n  H->>S: step_two()");
    let sf = check(&seq, "doc.md", &index, &dir);
    assert!(!sf.is_empty(), "sequence should ground to `handle`");
    assert!(sf.iter().all(|f| f.verdict == Verdict::Supported), "{sf:?}");

    let _ = fs::remove_dir_all(&dir);
}
