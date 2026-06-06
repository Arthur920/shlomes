use std::collections::HashSet;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use super::*;
use crate::claim::Provenance;
use crate::findings::Finding;

fn tmp(tag: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("shlomes-drift-{tag}-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn changed(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn lineage_dirty_when_anchor_changed() {
    let set = changed(&["src/api"]);
    assert!(is_dirty(&Provenance::modules(["src/api".into()]), &set));
    assert!(!is_dirty(&Provenance::modules(["src/db".into()]), &set));
    // Ungrounded claims are always dirty (can't be carried forward).
    assert!(is_dirty(&Provenance::default(), &set));
    // Path anchors match by file path.
    assert!(is_dirty(&Provenance::path("src/api"), &set));
}

#[test]
fn score_is_severity_weighted_supported_over_total() {
    // 2 supported (credit 2, total 2) + 1 contradicted (credit 0, total 3).
    let scored = vec![
        (Verdict::Supported, vec!["m".to_string()]),
        (Verdict::Supported, vec!["m".to_string()]),
        (Verdict::Contradicted, vec!["m".to_string()]),
    ];
    let s = compute_score(&scored, "abc");
    assert!((s.repo - (2.0 / 5.0)).abs() < 1e-9, "{}", s.repo);
    assert!((s.per_module["m"] - (2.0 / 5.0)).abs() < 1e-9);
}

#[test]
fn unverifiable_is_excluded_from_score() {
    let scored = vec![
        (Verdict::Supported, vec!["m".to_string()]),
        (Verdict::Unverifiable, vec!["m".to_string()]),
    ];
    let s = compute_score(&scored, "");
    assert!((s.repo - 1.0).abs() < 1e-9, "{}", s.repo);
}

#[test]
fn run_writes_ledger_and_score_and_treats_all_dirty_without_git() {
    let dir = tmp("write");
    let index = CodeIndex::default();
    let claims = vec![
        Finding::supported("a", "doc.md:1", Provenance::modules(["m".into()])),
        Finding::problem(Verdict::Stale, "b", "doc.md:2", "stale"),
    ];
    let opts = Options {
        write_ledger: true,
        ..Default::default()
    };
    let out = run(claims, &index, &dir, &opts);

    // Non-git dir => no carry-forward, everything dirty.
    assert_eq!(out.carried_forward, 0);
    assert_eq!(out.total_claims, 2);
    // One reportable problem (the stale); the supported claim is hidden.
    assert_eq!(out.findings.len(), 1);
    assert_eq!(out.findings[0].verdict, Verdict::Stale);

    // Artifacts written.
    assert!(dir.join(".shlomes/ledger.json").exists());
    assert!(dir.join(".shlomes/score.json").exists());
    let reloaded = Score::load(&dir).unwrap();
    // supported credit 1 / (1 + stale 2) = 1/3.
    assert!((reloaded.repo - (1.0 / 3.0)).abs() < 1e-9, "{}", reloaded.repo);
}

#[test]
fn path_claims_bucket_to_their_module_or_path() {
    use crate::code::symbol::{Facts, Span, Symbol, SymbolKind, Visibility};

    let sym = Symbol {
        qualified_name: "src/foo::bar".into(),
        name: "bar".into(),
        kind: SymbolKind::Function,
        visibility: Visibility::Public,
        module: "src/foo".into(),
        span: Span {
            path: "src/foo.rs".into(),
            start_line: 1,
            end_line: 2,
        },
        body_span: Span::zero(),
        signature: None,
        doc: None,
        facts: Facts::default(),
        calls: Vec::new(),
        members: Vec::new(),
    };
    let index = CodeIndex {
        symbols: vec![sym],
        ..Default::default()
    };
    // A code-file path resolves to its module, even when named by suffix only.
    assert_eq!(claim_modules(&Provenance::path("foo.rs"), &index), vec!["src/foo"]);
    assert_eq!(
        claim_modules(&Provenance::path("src/foo.rs"), &index),
        vec!["src/foo"]
    );
    // A non-code/unknown file falls back to bucketing by the path itself.
    assert_eq!(
        claim_modules(&Provenance::path("CLAUDE.md"), &index),
        vec!["CLAUDE.md"]
    );
}

#[test]
fn behavioral_drift_flag_fires_when_ledger_hash_is_stale() {
    use crate::claim::claim_id;
    use crate::code::facts;
    use crate::code::symbol::{Facts, Span, Symbol, SymbolKind, Visibility};
    use crate::drift::ledger::{ClaimRecord, Ledger};

    let dir = tmp("driftflag");

    // An index with one symbol carrying real facts (nonzero fingerprint).
    let sym = Symbol {
        qualified_name: "m::foo".into(),
        name: "foo".into(),
        kind: SymbolKind::Function,
        visibility: Visibility::Public,
        module: "m".into(),
        span: Span::zero(),
        body_span: Span::zero(),
        signature: Some("fn foo()".into()),
        doc: None,
        facts: Facts {
            constants: vec!["3".into()],
            ..Default::default()
        },
        calls: Vec::new(),
        members: Vec::new(),
    };
    let current_hash = facts::facts_hash(&sym.facts);
    assert_ne!(current_hash, 0);
    let index = CodeIndex {
        symbols: vec![sym],
        ..Default::default()
    };

    // A claim anchored to that symbol.
    let f = Finding::supported("calls `foo`", "doc.md:1", Provenance::symbol("m::foo"));
    let id = claim_id(&f.doc_path, &f.claim);

    // Seed a ledger whose stored fingerprint disagrees with the current one.
    let mut claims = BTreeMap::new();
    claims.insert(
        id.clone(),
        ClaimRecord {
            id,
            doc_ref: f.doc_path.clone(),
            provenance: f.provenance.clone(),
            facts: Facts::default(),
            facts_hash: current_hash ^ 0xdead_beef, // deliberately stale
            verdict: Verdict::Supported,
            commit: "old".into(),
        },
    );
    Ledger {
        version: Ledger::VERSION,
        claims,
    }
    .save(&dir)
    .unwrap();

    // Non-git dir → claim is dirty → its fingerprint is recompared.
    let out = run(vec![f], &index, &dir, &Options::default());
    assert!(
        out.findings
            .iter()
            .any(|x| x.verdict == Verdict::Unverifiable
                && x.detail.contains("Behavioral drift")),
        "{:?}",
        out.findings
    );
}

#[test]
fn coverage_undocumented_gap_lowers_its_module_score() {
    // A4 score integration: an Undocumented coverage gap is a scored claim. One
    // supported + one undocumented in module `m` => credit 1 / total 2.
    let scored = vec![
        (Verdict::Supported, vec!["m".to_string()]),
        (Verdict::Undocumented, vec!["m".to_string()]),
    ];
    let s = compute_score(&scored, "");
    assert!((s.repo - 0.5).abs() < 1e-9, "{}", s.repo);
    assert!((s.per_module["m"] - 0.5).abs() < 1e-9);
}

#[test]
fn regression_detected_against_committed_base_score() {
    let dir = tmp("regress");
    // A committed base score of a perfect repo.
    Score {
        repo: 1.0,
        per_module: Default::default(),
        commit: "base".into(),
    }
    .save(&dir)
    .unwrap();

    let index = CodeIndex::default();
    // This run is worse (a contradiction drags the score below 1.0).
    let claims = vec![
        Finding::supported("a", "doc.md:1", Provenance::default()),
        Finding::problem(Verdict::Contradicted, "b", "doc.md:2", "bad"),
    ];
    let opts = Options {
        fail_on_regression: true,
        ..Default::default()
    };
    let out = run(claims, &index, &dir, &opts);
    let (base, head) = out.regression.expect("should regress");
    assert!((base - 1.0).abs() < 1e-9);
    assert!(head < base);
}
