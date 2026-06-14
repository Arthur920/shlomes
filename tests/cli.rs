//! Black-box CLI tests: run the compiled `staleguard` binary against a temporary
//! fixture repo and assert on its stdout/exit. These exercise arg parsing →
//! scan → serialization end to end, which the inline unit tests don't cover.

use std::path::PathBuf;
use std::process::Command;

/// Path to the binary cargo built for this test run.
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_staleguard")
}

/// Create a unique temp dir with the given files (relative path, contents) and
/// return its path. Caller is responsible for cleanup via [`Fixture`].
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new(files: &[(&str, &str)]) -> Fixture {
        let dir = std::env::temp_dir().join(format!(
            "staleguard-it-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, contents) in files {
            let path = dir.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, contents).unwrap();
        }
        Fixture { dir }
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        Command::new(bin())
            .args(args)
            .arg(&self.dir)
            .output()
            .expect("failed to run staleguard binary")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[test]
fn version_reports_binary_name() {
    let out = Command::new(bin()).arg("--version").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("staleguard"),
        "--version should name the binary, got: {stdout}"
    );
}

#[test]
fn index_emits_json_symbols() {
    let fx = Fixture::new(&[("lib.rs", "pub fn greet() {}\n")]);
    let out = fx.run(&["index", "--format", "json"]);
    assert!(out.status.success(), "index should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("index output is valid JSON");
    let names: Vec<&str> = json["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["name"].as_str())
        .collect();
    assert!(names.contains(&"greet"), "expected `greet` in {names:?}");
}

#[test]
fn check_flags_undocumented_public_symbol() {
    // `helper` is public and documented nowhere with no internal callers — a
    // coverage gap Layer 1 should surface.
    let fx = Fixture::new(&[
        ("lib.rs", "pub fn greet() {}\npub fn helper() {}\n"),
        ("README.md", "# Demo\n\n`greet` greets the user.\n"),
    ]);
    let out = fx.run(&["check", "--format", "json"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("check output is valid JSON");
    let findings = json["findings"].as_array().expect("findings array");
    assert!(
        findings
            .iter()
            .any(|f| f["claim"].as_str().unwrap_or("").contains("helper")),
        "expected a finding about `helper`, got: {stdout}"
    );
}

#[test]
fn check_clean_repo_has_no_findings() {
    // A documented public symbol and nothing undocumented: no drift.
    let fx = Fixture::new(&[
        ("lib.rs", "pub fn greet() {}\n"),
        ("README.md", "# Demo\n\n`greet` greets the user.\n"),
    ]);
    let out = fx.run(&["check", "--format", "json"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("check output is valid JSON");
    assert_eq!(
        json["findings"].as_array().map(|a| a.len()),
        Some(0),
        "clean repo should yield no findings, got: {stdout}"
    );
}
