//! Shared finding type used by every layer.
//!
//! A `Finding` doubles as a *claim record*: it carries the doc assertion, the
//! [`Provenance`] it is anchored to in code, and a [`Verdict`]. A `Supported`
//! finding is a claim that verified — it is recorded in the drift ledger and
//! counted in the alignment score, but filtered out of the human report. Every
//! other verdict is a reportable problem.

use serde::{Deserialize, Serialize};

use crate::claim::Provenance;

/// Reporting severity of a finding, ordered `Note < Warning < Error`. Used to
/// gate output by a threshold (`--min-severity`) and to set the SARIF level.
/// The `Ord` derive relies on the variants being declared low-to-high.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Advisory; high-volume (e.g. undocumented surface). Off by gating thresholds.
    Note,
    /// Couldn't be confirmed either way.
    Warning,
    /// Provably wrong: a broken reference or a contradicted claim.
    Error,
}

impl Severity {
    /// SARIF level string for this severity.
    pub fn as_sarif_level(self) -> &'static str {
        match self {
            Severity::Note => "note",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// doc claim disagrees with code
    Contradicted,
    /// doc refers to something that no longer exists
    Stale,
    /// could not gather evidence either way
    Unverifiable,
    /// public code surface that no doc references (code -> doc gap)
    Undocumented,
    /// claim backed by code (not reported by default)
    Supported,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Contradicted => "contradicted",
            Verdict::Stale => "stale",
            Verdict::Unverifiable => "unverifiable",
            Verdict::Undocumented => "undocumented",
            Verdict::Supported => "supported",
        }
    }

    /// Whether this verdict is shown in the human report. `Supported` claims are
    /// ledgered and scored but not reported.
    pub fn is_reportable(&self) -> bool {
        !matches!(self, Verdict::Supported)
    }

    /// Reporting severity. `Supported` has no severity (it is never reported);
    /// it maps to `Note` so it is never dropped by a threshold filter, which only
    /// ever touches reportable findings.
    pub fn severity(&self) -> Severity {
        match self {
            Verdict::Contradicted | Verdict::Stale => Severity::Error,
            Verdict::Unverifiable => Severity::Warning,
            Verdict::Undocumented => Severity::Note,
            Verdict::Supported => Severity::Note,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub verdict: Verdict,
    /// the doc assertion under test
    pub claim: String,
    /// where the claim came from (path:line)
    pub doc_path: String,
    /// human-readable explanation
    pub detail: String,
    /// 1 deterministic | 2 retrieval | 3 llm
    pub layer: u8,
    /// supporting / conflicting code references
    pub code_refs: Vec<String>,
    /// what the claim is anchored to in code (for drift lineage). Defaults to
    /// empty so older serialized findings still deserialize.
    #[serde(default)]
    pub provenance: Provenance,
}

impl Finding {
    /// A reportable problem (non-`Supported`). `provenance`/`code_refs` start
    /// empty; attach them with [`Finding::anchored`] / [`Finding::with_refs`].
    pub fn problem(
        verdict: Verdict,
        claim: impl Into<String>,
        doc_path: impl Into<String>,
        detail: impl Into<String>,
    ) -> Finding {
        Finding {
            verdict,
            claim: claim.into(),
            doc_path: doc_path.into(),
            detail: detail.into(),
            layer: 1,
            code_refs: Vec::new(),
            provenance: Provenance::default(),
        }
    }

    /// A claim that verified. Recorded + scored, never reported.
    pub fn supported(
        claim: impl Into<String>,
        doc_path: impl Into<String>,
        provenance: Provenance,
    ) -> Finding {
        Finding {
            verdict: Verdict::Supported,
            claim: claim.into(),
            doc_path: doc_path.into(),
            detail: String::new(),
            layer: 1,
            code_refs: Vec::new(),
            provenance,
        }
    }

    /// Attach provenance (builder style).
    pub fn anchored(mut self, provenance: Provenance) -> Finding {
        self.provenance = provenance;
        self
    }

    /// Attach code references (builder style).
    pub fn with_refs(mut self, code_refs: Vec<String>) -> Finding {
        self.code_refs = code_refs;
        self
    }
}
