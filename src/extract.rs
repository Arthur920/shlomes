//! Pull verifiable claims out of markdown docs.
//!
//! For now this only surfaces the deterministically checkable claims (paths).
//! Layer-3 free-text claim extraction (LLM) plugs in alongside these.

use std::sync::OnceLock;

use regex::Regex;

#[derive(Debug, Clone)]
pub struct PathClaim {
    /// the quoted token, e.g. "src/index.ts"
    pub raw: String,
    /// markdown file it appeared in
    pub doc_path: String,
    pub line: usize,
    /// True when the path is named in a context that asserts its *absence* or
    /// *former* state — a deletion note (`**Delete** … no longer exists`) or a
    /// migration "old → new" row. Such references must not yield stale-path
    /// findings: the missing file confirms the doc rather than contradicting it.
    pub historical: bool,
}

/// `backtick-quoted` tokens that look like a relative file or dir path.
fn path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"`([\w./\-]+/[\w./\-]+|[\w\-]+\.[\w]{1,5})`").unwrap())
}

/// File extensions a real on-disk path actually carries. The regex above also
/// matches dotted *code* references (`typing.Deque`, `dict.get`, `Config.extra`)
/// and prose slashes (`self/cls`, `include/exclude`, `read/write`); gating on a
/// real extension keeps those out of the path check, where they only ever
/// produced false "path does not exist" findings. Dotted symbols are the
/// symbol-resolver's job, not the filesystem's.
const PATH_EXTS: &[&str] = &[
    "py",
    "pyi",
    "pyx",
    "rs",
    "js",
    "mjs",
    "cjs",
    "jsx",
    "ts",
    "tsx",
    "go",
    "java",
    "kt",
    "rb",
    "c",
    "h",
    "cc",
    "cpp",
    "cxx",
    "hpp",
    "cs",
    "php",
    "swift",
    "scala",
    "clj",
    "ex",
    "exs",
    "erl",
    "hs",
    "ml",
    "fs",
    "dart",
    "lua",
    "pl",
    "pm",
    "r",
    "jl",
    "nim",
    "zig",
    "md",
    "markdown",
    "rst",
    "adoc",
    "txt",
    "toml",
    "yaml",
    "yml",
    "json",
    "json5",
    "jsonc",
    "cfg",
    "ini",
    "conf",
    "lock",
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "bat",
    "cmd",
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    "sql",
    "xml",
    "csv",
    "tsv",
    "proto",
    "graphql",
    "gql",
    "vue",
    "svelte",
    "mk",
    "gradle",
    "properties",
    "dot",
    "gv",
    "ipynb",
    "tf",
    "hcl",
    "env",
    "rake",
    "gemspec",
    "podspec",
];

/// Extensionless filenames that are nonetheless real, well-known repo files.
const PATH_FILENAMES: &[&str] = &[
    "makefile",
    "gnumakefile",
    "dockerfile",
    "license",
    "readme",
    "changelog",
    "procfile",
    "gemfile",
    "rakefile",
    "justfile",
    "vagrantfile",
    "brewfile",
];

/// Slashless `word.ext` tokens whose extension collides with code attribute
/// access (`express.json`, `app.css`, `query.sql`) are ambiguous; only these
/// well-known root manifests are treated as paths when named bare. Anything with
/// a directory segment (`config/app.json`) is unambiguous and handled directly.
const CANONICAL_FILES: &[&str] = &[
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "jsconfig.json",
    "cargo.toml",
    "cargo.lock",
    "pyproject.toml",
    "composer.json",
    "composer.lock",
    "deno.json",
    "deno.jsonc",
    "go.mod",
    "go.sum",
    "angular.json",
    "nx.json",
];

/// Does this regex match name an actual repo path (vs. a dotted code reference or
/// a prose `a/b` slash)? True only when its final segment carries a known file
/// extension or is a known extensionless filename.
fn looks_like_path(token: &str) -> bool {
    let has_slash = token.contains('/');
    let last = token.rsplit('/').next().unwrap_or(token);
    let lower = last.to_ascii_lowercase();
    if let Some((_, ext)) = last.rsplit_once('.') {
        if PATH_EXTS.contains(&ext.to_ascii_lowercase().as_str()) {
            // A slashless `word.ext` is ambiguous between a file and a dotted code
            // reference (`express.json`, `app.css`). Treat it as a path only when
            // it carries a directory or is a well-known root manifest; otherwise
            // defer to the symbol resolver — guessing here only ever produced
            // false "path does not exist" findings.
            return has_slash || CANONICAL_FILES.contains(&lower.as_str());
        }
    }
    PATH_FILENAMES.contains(&lower.as_str())
}

/// Cue words that mark a path as deleted, renamed, or otherwise historical.
fn historical_cue_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(delete[ds]?|remove[ds]?|removal|renamed?|moved|former(ly)?|previously|old path|original plan|as-built|replaced?|no longer|used to|deprecated|legacy)\b",
        )
        .unwrap()
    })
}

/// Lines of preceding context scanned (plus the claim's own line) for a cue. A
/// small window so a `**Delete**` heading or an `Original → As-built` table
/// header still covers the rows beneath it without bleeding into unrelated prose.
const CONTEXT_WINDOW: usize = 4;

/// Is the path on line `idx` named in a deletion/migration context?
fn in_historical_context(lines: &[&str], idx: usize) -> bool {
    let start = idx.saturating_sub(CONTEXT_WINDOW);
    lines[start..=idx]
        .iter()
        .any(|l| historical_cue_re().is_match(l))
}

/// Find backtick-quoted tokens that look like paths the repo should contain.
pub fn extract_path_claims(markdown: &str, doc_path: &str) -> Vec<PathClaim> {
    let mut claims = Vec::new();
    let lines: Vec<&str> = markdown.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        for cap in path_re().captures_iter(line) {
            let token = &cap[1];
            // Skip obvious non-paths (URLs, version specifiers, globs).
            if token.contains("://") || token.starts_with('*') || token.ends_with("/*") {
                continue;
            }
            if !looks_like_path(token) {
                continue;
            }
            claims.push(PathClaim {
                raw: token.to_string(),
                doc_path: doc_path.to_string(),
                line: i + 1,
                historical: in_historical_context(&lines, i),
            });
        }
    }
    claims
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raws(md: &str) -> Vec<String> {
        extract_path_claims(md, "README.md")
            .into_iter()
            .map(|c| c.raw)
            .collect()
    }

    #[test]
    fn real_paths_are_claimed() {
        let got = raws("See `src/main.py`, `docs/usage/custom.md`, and `pydantic-core/Makefile`.");
        assert!(got.contains(&"src/main.py".to_string()));
        assert!(got.contains(&"docs/usage/custom.md".to_string()));
        assert!(got.contains(&"pydantic-core/Makefile".to_string()));
    }

    #[test]
    fn dotted_symbols_are_not_paths() {
        // `module.Symbol` / `Type.method` refs are the symbol resolver's job;
        // they must never become "path does not exist" findings.
        let got = raws("`typing.Deque`, `dict.get`, `json.dumps`, `Config.extra`, `pydantic.v1`.");
        assert!(got.is_empty(), "dotted code refs leaked as paths: {got:?}");
    }

    #[test]
    fn prose_slashes_are_not_paths() {
        let got = raws("Either `self/cls`, `include/exclude`, or `BaseModel/RootModel`.");
        assert!(got.is_empty(), "prose slashes leaked as paths: {got:?}");
    }

    #[test]
    fn slashless_code_refs_with_path_extensions_are_not_paths() {
        // Suffix collides with a real extension but it's a code reference, not a
        // file: `express.json` is an Express API, `app.css` a styled-component, etc.
        let got = raws("Use `express.json`, `app.css`, `query.sql`, and `router.use`.");
        assert!(
            got.is_empty(),
            "code refs with path-like extensions leaked as paths: {got:?}"
        );
    }

    #[test]
    fn bare_root_manifests_and_dir_paths_are_still_claimed() {
        // Canonical bare manifests and any token with a directory remain paths.
        let got = raws("See `package.json`, `tsconfig.json`, and `src/config/app.json`.");
        assert!(got.contains(&"package.json".to_string()));
        assert!(got.contains(&"tsconfig.json".to_string()));
        assert!(got.contains(&"src/config/app.json".to_string()));
    }

    #[test]
    fn deletion_and_migration_context_marks_historical() {
        let md = "\
**Delete**

- `Backend/src/http/async-handler.ts`

The file `Backend/src/http/async-handler.ts` no longer exists.

| Original plan | As-built |
| --- | --- |
| `src/lib/query-client.ts` | `src/common/query/query-client.ts` |
";
        let claims = extract_path_claims(md, "PLAN.md");
        let hist: Vec<&str> = claims
            .iter()
            .filter(|c| c.historical)
            .map(|c| c.raw.as_str())
            .collect();
        assert!(hist.contains(&"Backend/src/http/async-handler.ts"));
        assert!(hist.contains(&"src/lib/query-client.ts"));
    }

    #[test]
    fn ordinary_path_is_not_historical() {
        let claims = extract_path_claims("The entry point is `src/main.ts`.", "README.md");
        assert!(claims.iter().all(|c| !c.historical));
    }
}
