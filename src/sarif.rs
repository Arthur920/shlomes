//! SARIF 2.1.0 rendering for `staleguard check`.
//!
//! SARIF is the format GitHub's code-scanning UI ingests: upload the output with
//! `github/codeql-action/upload-sarif` and every finding shows up as an inline
//! annotation on the offending doc line, plus an entry in the Security tab. It
//! is the adoption path for using Staleguard as a CI gate without writing a
//! custom JSON parser.
//!
//! Only `check` produces SARIF — it is the doc-vs-code findings command. The
//! other subcommands keep `text`/`json`.

use serde_json::{json, Value};

use crate::findings::{Finding, Verdict};

/// SARIF severity level for a verdict, from the shared [`Verdict::severity`].
fn level(verdict: Verdict) -> &'static str {
    verdict.severity().as_sarif_level()
}

/// Split a `path:line` doc reference into its components. A missing/invalid line
/// suffix falls back to line 1 so the result still anchors to the file.
fn split_ref(doc_ref: &str) -> (&str, i64) {
    match doc_ref.rsplit_once(':') {
        Some((path, line)) => match line.parse::<i64>() {
            Ok(n) if n >= 1 => (path, n),
            _ => (doc_ref, 1),
        },
        None => (doc_ref, 1),
    }
}

/// Build one SARIF result object for a reportable finding.
fn result(f: &Finding) -> Value {
    let (path, line) = split_ref(&f.doc_path);
    json!({
        "ruleId": f.verdict.as_str(),
        "level": level(f.verdict),
        "message": { "text": f.detail },
        "locations": [{
            "physicalLocation": {
                "artifactLocation": { "uri": path },
                "region": { "startLine": line }
            }
        }],
        "properties": { "claim": f.claim, "layer": f.layer }
    })
}

/// One SARIF rule descriptor per verdict category Staleguard can emit.
fn rule_descriptors() -> Value {
    let rules = [
        (
            Verdict::Stale,
            "A doc references a path, command, symbol, env var, or flag that no longer exists in the code.",
        ),
        (
            Verdict::Contradicted,
            "A doc claim disagrees with what the code actually does.",
        ),
        (
            Verdict::Unverifiable,
            "A doc claim could not be confirmed or refuted from the code.",
        ),
        (
            Verdict::Undocumented,
            "Public code surface that no doc describes (code -> doc gap).",
        ),
    ];
    Value::Array(
        rules
            .iter()
            .map(|(v, desc)| {
                json!({
                    "id": v.as_str(),
                    "name": v.as_str(),
                    "shortDescription": { "text": *desc },
                    "defaultConfiguration": { "level": level(*v) }
                })
            })
            .collect(),
    )
}

/// Render reportable findings as a SARIF 2.1.0 log document.
pub fn render(findings: &[Finding]) -> Value {
    let results: Vec<Value> = findings
        .iter()
        .filter(|f| f.verdict.is_reportable())
        .map(result)
        .collect();
    json!({
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "Staleguard",
                    "informationUri": "https://github.com/Arthur920/Staleguard",
                    "version": env!("CARGO_PKG_VERSION"),
                    "rules": rule_descriptors()
                }
            },
            "results": results
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_path_and_line() {
        assert_eq!(split_ref("docs/a.md:42"), ("docs/a.md", 42));
        assert_eq!(split_ref("README.md"), ("README.md", 1));
        // A non-positive line number is invalid: keep the whole ref, line 1.
        assert_eq!(split_ref("a:b.md:0"), ("a:b.md:0", 1));
    }

    #[test]
    fn supported_findings_are_omitted() {
        let findings = vec![
            Finding::problem(Verdict::Stale, "c", "README.md:3", "broken path"),
            Finding::supported("ok", "README.md:5", Default::default()),
        ];
        let sarif = render(&findings);
        let results = sarif["runs"][0]["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["ruleId"], "stale");
        assert_eq!(results[0]["level"], "error");
        assert_eq!(
            results[0]["locations"][0]["physicalLocation"]["region"]["startLine"],
            3
        );
    }

    #[test]
    fn shape_is_valid_sarif() {
        let sarif = render(&[]);
        assert_eq!(sarif["version"], "2.1.0");
        assert!(sarif["runs"][0]["tool"]["driver"]["rules"]
            .as_array()
            .is_some_and(|r| r.len() == 4));
    }
}
