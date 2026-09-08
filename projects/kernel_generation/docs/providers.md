# Provider and model configuration

Providers are JSON files in `~/.kernelguy/providers/`. On first launch the defaults
from `src/default_providers/` are copied there, and existing files are never
overwritten, so your edits survive upgrades. The flip side: a model added to a
shipped default does *not* reach a machine that has already launched — delete the
file to re-seed it.

A provider's name is its filename stem, so nothing inside the file can disagree with
what it is called. `--provider` names it without the `.json`, and there is no
default, so which file answered is never implicit.

## File format

Each file declares a `base_url`, and one endpoint per wire protocol carrying the
models reachable over it and the credential it takes:

```json
{
  "base_url": "https://<host>/api/v2/cortex/v1",
  "headers": { "X-SNOWFLAKE-APPLICATION": "kernelguy" },
  "endpoints": [
    {
      "protocol": "anthropic-messages",
      "auth": { "command": { "binary": "cortex-oauth-helper", "args": ["token"] } },
      "headers": { "anthropic-beta": "context-1m-2025-08-07" },
      "models": [{ "id": "claude-sonnet-5", "context_size": 1000000 }]
    }
  ]
}
```

Every `*.json` in the directory is read at startup. A malformed one is fatal rather
than skipped, because skipping would present as "that model doesn't exist".

`base_url` carries no protocol path. Each protocol appends its conventional one:
`anthropic-messages` → `/messages`, `openai-responses` → `/responses`,
`openai-chat-completions` → `/chat/completions`. That convention holds for Cortex,
OpenAI direct, Anthropic direct and Ollama. It does not hold for Azure (deployment
name in the path plus an `api-version` query) or Bedrock (model in the path), and
there is no per-endpoint path override.

## Model resolution

`--model` must name a model the chosen provider lists; an unknown slug is an error
that prints what *is* served. The listing endpoint determines both the protocol used
and the context window, so adding a model or retargeting a host is a file edit, not
a rebuild.

When several endpoints of one provider list the same id, the lowest-ordered protocol
wins — `openai-responses`, then `anthropic-messages`, then
`openai-chat-completions`. That makes the choice a property of the protocol set
rather than of the order endpoints happen to appear in the file. Pass
`--protocol-override` to pin one instead.

## Auth

`auth` sits on the endpoint rather than the provider, because a host can serve two
protocols behind different credentials. Three forms:

| form | behaviour |
| --- | --- |
| `{"command": {"binary": "…", "args": […]}}` | stdout is the key. Spawned directly — no shell, so nothing word-splits or expands the arguments. Cached, re-run when a request fails, never run in parallel. |
| `{"env_var": "NAME"}` | read and cached once at startup, so changing the variable mid-run has no effect. |
| `{"key": "…"}` | a constant, in cleartext in the file. Convenient for a local server. |

A broken credential fails startup rather than mid-run: an unset or empty `env_var`,
or a `command` with no binary or one that fails its first invocation, refuses the
run.

`base_url` and `auth` are independent and nothing makes them agree. If the auth
command authenticates against a different account than `base_url` points at,
requests fail as an unexplained 401. Keeping them consistent is the operator's job.

## Custom headers

Both `headers` blocks are optional — omit the key entirely, rather than writing
`null`, which is a parse error. The provider-level one is sent to every endpoint of
that host; an endpoint's own entries are layered on top, which is where anything
protocol-specific belongs. Names are compared case-insensitively, as HTTP does, so
an endpoint's `x-foo` replaces the provider's `X-Foo` rather than adding a second
header. One value per name.

Unknown keys anywhere in a provider file are rejected at startup, naming the
offending field and listing the valid ones. That matters most for an optional
block: without it, a capitalised `Headers` or a singular `header` would parse
happily and simply send nothing.

Headers kernelguy or hyper sets itself are also rejected at startup, with an error
naming the key and why it cannot stand. `Authorization` and `Content-Type` would be
**silently discarded** — the first comes from `auth`, the second is set per request
with the JSON body, and client defaults fill only headers the request left vacant.
`Content-Length`, `Transfer-Encoding`, `Host` and `Accept-Encoding` are worse: hyper
honours a caller-set value, so they corrupt the exchange rather than being ignored.
A malformed name or value is a startup error naming the key too.

**Header values sit in cleartext in the provider file**, exactly as `auth`'s
`{"key": …}` form does. For a credential prefer `{"command": …}` or
`{"env_var": …}`, which keep the secret out of the file altogether.

Redirects are disabled on the model client. reqwest would otherwise follow up to
ten, dropping only `Authorization`, `Cookie`, `Proxy-Authorization` and
`WWW-Authenticate` when the origin changes — a configured header is not on that
list, so it would be resent to whatever host the `Location` named. A `3xx` from a
completions endpoint therefore surfaces as that status instead of being followed.

Two limits worth knowing. A `--supervised` model reuses the **main** model's
endpoint, so it also gets that endpoint's headers even if the supervisor's slug is
listed elsewhere. And these headers apply only to model requests; the `view` tool's
fetches of third-party image URLs deliberately do not carry them.
