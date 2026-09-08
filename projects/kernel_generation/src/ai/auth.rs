//! Credential supply for the completion clients, separated from the wire
//! protocol because the two vary independently: one HTTP API, either a static
//! key or a rotating token behind it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use tokio::sync::Mutex;

use crate::ai::CompletionError;

/// Per-request source of the `Authorization` header value.
///
/// Supplies the `Authorization` header value for each request. This lets the
/// completion clients use a rotating credential — e.g. an access token minted by
/// an external helper and renewed as it nears expiry — instead of a header fixed
/// at construction time.
///
/// Implementors return the full header value, e.g. `Bearer <token>`. A failed
/// renewal maps to [`CompletionError`] so the harness's retry policy applies
/// (network blips → `Transient`); the client adds the header per request.
pub trait AuthProvider: Send + Sync {
    /// The `Authorization` header value.
    ///
    /// `force_refresh` means the *server* just rejected the current credential — a
    /// 401 "token expired" that beat the local expiry check, from clock skew or
    /// propagation delay. Renew even though the local cache still considers the
    /// credential valid, so a single stale 401 cannot kill a long run. Static
    /// credentials ignore it: there is nothing to renew, so a 401 there is a
    /// genuine bad credential.
    fn authorization(&self, force_refresh: bool) -> BoxFuture<'_, Result<String, CompletionError>>;
}

/// Why a credential could not be obtained.
///
/// Typed rather than a `String` so callers can tell a *permanent*
/// misconfiguration ([`Self::NoBinary`], [`Self::EnvVarUnset`]) from a *transient*
/// helper failure, and so the underlying [`std::io::Error`] survives as a
/// `source` instead of being flattened into prose.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("auth command has no binary")]
    NoBinary,
    #[error("{var} is not set")]
    EnvVarUnset { var: String },
    #[error("{var} is set but empty")]
    EnvVarEmpty { var: String },
    #[error("could not run `{binary}`: {source}")]
    Spawn {
        binary: String,
        #[source]
        source: std::io::Error,
    },
    #[error("`{binary}` timed out after {}s", after.as_secs())]
    Timeout { binary: String, after: Duration },
    #[error("`{binary}` exited with {status}: {stderr}")]
    Exit {
        binary: String,
        status: std::process::ExitStatus,
        stderr: String,
    },
    #[error("`{binary}` succeeded but printed no token")]
    NoOutput { binary: String },
}

impl From<AuthError> for CompletionError {
    /// Auth failures are `Transient` so the harness's retry covers a blip in the
    /// helper. A genuinely broken one keeps failing and surfaces once the retry
    /// budget is spent; the permanent variants are caught at startup by
    /// `AuthMethod::resolve` before any request goes out.
    fn from(e: AuthError) -> Self {
        Self::Transient(e.to_string())
    }
}

/// Renew slightly before the assumed expiry to absorb clock skew + request
/// latency, so a token isn't used in the moment it lapses.
const EXPIRY_SKEW: Duration = Duration::from_mins(1);

/// Assumed lifetime of a command-minted access token. The helper prints only the
/// token, never its expiry, so there is no real lifetime to read off the wire; we
/// renew proactively on this cadence instead (and reactively on a server 401, see
/// [`AuthProvider::force_refresh`]). Re-running the helper is cheap — it reuses
/// its own cached credentials — so renewing early costs almost nothing.
///
/// MUST exceed [`EXPIRY_SKEW`] or a freshly minted token reads as already
/// expired and every single request re-runs the command. Guarded by
/// `assumed_ttl_exceeds_expiry_skew`.
const ASSUMED_TTL: Duration = Duration::from_mins(5);

/// Age at which a token is renewed: [`ASSUMED_TTL`] less the skew, precomputed so
/// the hot path does no arithmetic. Saturates to zero if the two constants are
/// ever misordered, which would renew on *every* request — the failure mode
/// `assumed_ttl_exceeds_expiry_skew` exists to catch.
const RENEW_AFTER: Duration = ASSUMED_TTL.saturating_sub(EXPIRY_SKEW);

/// Dedup window for server-driven forced renewals. When several concurrent
/// requests all get a 401 for the same just-expired token, only the first runs
/// the command; the rest reuse the token it fetched instead of a spawn storm.
const FORCED_REFRESH_DEDUP: Duration = Duration::from_secs(5);

/// Cap on how long we wait for the command to print a token. The helper is
/// non-interactive and returns promptly; the timeout only guards against a hang
/// wedging a run (the failure mode that cost `run_1784926141` 180s).
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);

/// A credential obtained by running an external command that prints it to
/// stdout — e.g. `cortex-oauth-helper token`, which mints a currently-valid
/// access token, refreshes silently via its own cached credentials, and never
/// opens a browser, so it works headless.
///
/// The command is opaque: it is run as given and its stdout taken as the token.
/// Nothing here parses it or adds arguments, so the helper owns credential
/// storage, refreshing, and any account/role discovery. This type owns only the
/// in-memory token and the renewal cadence. Cheap to clone (one shared inner
/// state behind an `Arc`) and implements [`AuthProvider`], so it can drive the
/// completion clients directly.
#[derive(Clone)]
pub struct CommandAuth {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    /// Program to spawn. Present by construction — there is no empty-argv case.
    binary: String,
    args: Vec<String>,
    token: String,
    /// When the current token was minted; it is treated as valid for
    /// [`ASSUMED_TTL`] from this point.
    minted_at: Instant,
    /// Timestamp of the last forced renewal, for the [`FORCED_REFRESH_DEDUP`]
    /// concurrent-401 dedup. `None` until one runs.
    last_forced: Option<Instant>,
}

impl CommandAuth {
    /// Run `binary args...` once up front, so a missing or broken helper fails
    /// during startup rather than mid-run.
    ///
    /// The program is spawned directly — no shell — so nothing here word-splits,
    /// glob-expands, or interprets the arguments, and an argument containing spaces
    /// needs no quoting. Its stdin is closed, so it can never block waiting on
    /// input.
    ///
    /// # Errors
    ///
    /// Returns an error if `binary` is blank, or if the first invocation fails
    /// (absent, times out after [`COMMAND_TIMEOUT`], exits non-zero, or prints
    /// nothing).
    pub async fn new(binary: &str, args: Vec<String>) -> Result<Self, AuthError> {
        let binary = binary.trim().to_string();
        if binary.is_empty() {
            return Err(AuthError::NoBinary);
        }
        let token = run(&binary, &args).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                binary,
                args,
                token,
                minted_at: Instant::now(),
                last_forced: None,
            })),
        })
    }

    /// Return a currently-valid token, re-running the command first if needed.
    ///
    /// Normally "needed" means the cached token has reached [`RENEW_AFTER`]. With
    /// `force_refresh` it means the server rejected the token regardless of what
    /// the local clock thinks — the contract's "should be reran when a request
    /// fails". Forced renewals are de-duped within [`FORCED_REFRESH_DEDUP`] so a
    /// wave of concurrent 401s for the same token runs the command once.
    ///
    /// The contract requires the command never run in parallel; the lock is what
    /// guarantees it, and it also means a burst of concurrent requests triggers at
    /// most one renewal.
    ///
    /// # Errors
    ///
    /// Returns an error if a renewal is warranted and the command fails. The
    /// failure propagates so the harness's retry can attempt a fresh renewal.
    pub async fn token(&self, force_refresh: bool) -> Result<String, AuthError> {
        let mut inner = self.inner.lock().await;
        let renew = if force_refresh {
            // A concurrent request may already have renewed for this same 401 wave.
            inner.last_forced.is_none_or(|t| t.elapsed() >= FORCED_REFRESH_DEDUP)
        } else {
            expired(inner.minted_at)
        };
        if renew {
            if force_refresh {
                eprintln!(
                    "kernelguy: token rejected by server (401); renewing via `{}`...",
                    inner.binary
                );
            } else {
                eprintln!("kernelguy: renewing access token via `{}`...", inner.binary);
            }
            let token = run(&inner.binary, &inner.args).await?;
            inner.token = token;
            inner.minted_at = Instant::now();
            if force_refresh {
                inner.last_forced = Some(Instant::now());
            }
        }
        Ok(inner.token.clone())
    }
}

/// Whether a token minted at `minted_at` should be renewed — it has reached
/// [`RENEW_AFTER`]. Split out so the cadence is unit-testable without running a
/// command.
fn expired(minted_at: Instant) -> bool {
    minted_at.elapsed() >= RENEW_AFTER
}

/// Spawn `binary args...` directly (no shell) and return its trimmed stdout.
async fn run(binary: &str, args: &[String]) -> Result<String, AuthError> {
    let mut cmd = tokio::process::Command::new(binary);
    cmd.args(args);
    // Close stdin so the command can never block waiting on input.
    cmd.stdin(std::process::Stdio::null());

    let output = tokio::time::timeout(COMMAND_TIMEOUT, cmd.output())
        .await
        .map_err(|_| AuthError::Timeout {
            binary: binary.to_string(),
            after: COMMAND_TIMEOUT,
        })?
        .map_err(|source| AuthError::Spawn {
            binary: binary.to_string(),
            source,
        })?;
    if !output.status.success() {
        return Err(AuthError::Exit {
            binary: binary.to_string(),
            status: output.status,
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(AuthError::NoOutput {
            binary: binary.to_string(),
        });
    }
    Ok(token)
}

impl AuthProvider for CommandAuth {
    fn authorization(&self, force_refresh: bool) -> BoxFuture<'_, Result<String, CompletionError>> {
        Box::pin(async move {
            self.token(force_refresh)
                .await
                .map(|token| format!("Bearer {token}"))
                .map_err(CompletionError::from)
        })
    }
}

/// A credential fixed at construction — an API key or personal access token.
///
/// `force_refresh` is ignored: there is nothing to renew, so a 401 here is a
/// genuine bad credential rather than an expiry.
pub struct StaticAuth {
    header: String,
}

impl StaticAuth {
    /// Wrap `token` as a `Bearer` header value.
    #[must_use]
    pub fn bearer(token: impl AsRef<str>) -> Self {
        Self {
            header: format!("Bearer {}", token.as_ref()),
        }
    }
}

impl AuthProvider for StaticAuth {
    fn authorization(&self, _force_refresh: bool) -> BoxFuture<'_, Result<String, CompletionError>> {
        let header = self.header.clone();
        Box::pin(async move { Ok(header) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant that replaces the old refresh-storm cooldown: because we
    /// assign the lifetime ourselves (the helper reports none), a token can only
    /// read as born-expired if these two constants are misordered. If that ever
    /// happens, every request re-runs the command.
    #[test]
    fn assumed_ttl_exceeds_expiry_skew() {
        assert!(ASSUMED_TTL > EXPIRY_SKEW);
    }

    #[test]
    fn a_freshly_minted_token_is_not_expired() {
        assert!(!expired(Instant::now()));
    }

    #[test]
    fn a_token_at_the_renewal_age_is_expired() {
        let Some(old) = Instant::now().checked_sub(RENEW_AFTER) else {
            return;
        };
        assert!(expired(old));
    }

    #[test]
    fn a_token_past_its_assumed_ttl_is_expired() {
        let Some(old) = Instant::now().checked_sub(ASSUMED_TTL) else {
            return;
        };
        assert!(expired(old));
    }

    #[tokio::test]
    async fn a_blank_binary_is_rejected() {
        assert!(CommandAuth::new("   ", vec![]).await.is_err());
    }

    #[tokio::test]
    async fn a_command_that_prints_nothing_is_rejected() {
        assert!(run("true", &[]).await.is_err());
    }

    #[tokio::test]
    async fn a_failing_command_surfaces_its_stderr() {
        let err = run("sh", &["-c".to_string(), "echo boom >&2; exit 3".to_string()])
            .await
            .unwrap_or_else(|e| e.to_string());
        assert!(err.contains("exited with"), "{err}");
        assert!(err.contains("boom"), "{err}");
    }

    #[tokio::test]
    async fn an_absent_helper_is_a_spawn_error() {
        // The real-world failure: the helper isn't installed on this box. Spawning
        // directly means this is an OS spawn error, not a shell's 127 exit.
        let err = run("definitely-not-a-real-binary-xyz", &[])
            .await
            .unwrap_or_else(|e| e.to_string());
        assert!(err.contains("could not run"), "{err}");
    }

    #[tokio::test]
    async fn a_command_stdout_is_trimmed() {
        let token = run("echo", &["  tok  ".to_string()]).await.unwrap_or_default();
        assert_eq!(token, "tok");
    }

    /// No shell means no word-splitting: an argument containing spaces arrives as
    /// ONE argument, needing no quoting. A `bash -c` implementation would split it.
    #[tokio::test]
    async fn an_argument_containing_spaces_is_passed_as_one_arg() {
        let token = run("echo", &["-n".to_string(), "a b c".to_string()])
            .await
            .unwrap_or_default();
        assert_eq!(token, "a b c");
    }
}
