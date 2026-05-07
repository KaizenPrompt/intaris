# Quick Start

Get Intaris running and connected to your first AI agent in 5 minutes.

## Step 1: Start Intaris

Intaris needs an OpenAI-compatible API key for safety evaluation. It picks up `LLM_API_KEY` from your environment automatically.

**uvx (recommended):**

```bash
LLM_API_KEY=sk-your-key uvx intaris
```

**Docker:**

```bash
LLM_API_KEY=sk-your-key docker compose up -d
```

**pip:**

```bash
pip install intaris
LLM_API_KEY=sk-your-key intaris
```

Intaris starts on `http://localhost:8060`. Open `http://localhost:8060/ui` in your browser to see the management dashboard.

## Step 2: Connect Your Client

### OpenCode (Plugin)

The fastest way to get started with [OpenCode](https://opencode.ai):

```bash
export INTARIS_URL=http://localhost:8060
cp integrations/opencode/intaris.ts ~/.config/opencode/plugins/
```

Run OpenCode — every tool call is now evaluated by Intaris before execution.

### Claude Code (Hooks)

For [Claude Code](https://docs.anthropic.com/en/docs/claude-code):

```bash
export INTARIS_URL=http://localhost:8060
mkdir -p ~/.claude/scripts
cp integrations/claude-code/scripts/*.sh ~/.claude/scripts/
chmod +x ~/.claude/scripts/*.sh
```

Merge `integrations/claude-code/hooks.json` into your `~/.claude/settings.json`. See the [Claude Code Guide](clients/claude-code.md) for details.

### OpenClaw (Extension)

For [OpenClaw](https://github.com/fpytloun/openclaw/tree/v2026.3.13) (version `v2026.3.13`):

The Intaris extension is built into the OpenClaw repository at `extensions/intaris/`. It is loaded automatically when present. Configure via OpenClaw's settings UI or environment variables:

```bash
export INTARIS_URL=http://localhost:8060
export INTARIS_API_KEY=your-api-key
```

The extension evaluates every tool call, manages session lifecycle, and optionally proxies upstream MCP tools. See the [OpenClaw Guide](clients/openclaw.md) for details.

### Any MCP Client (Proxy)

For any MCP-compatible client, point it at Intaris's `/mcp` endpoint:

```json
{
  "mcpServers": {
    "intaris": {
      "type": "streamable-http",
      "url": "http://localhost:8060/mcp"
    }
  }
}
```

Then configure upstream MCP servers in Intaris via the UI or REST API. See the [MCP Proxy Guide](mcp-proxy.md).

## Step 3: Try It

Trigger a tool call from your agent. You should see:

- **Management UI** (`http://localhost:8060/ui`) — sessions, evaluations, and audit records appear in real time
- **Server logs** — evaluation decisions logged with tool name, decision, and latency
- **Agent output** — approved calls execute normally; denied/escalated calls show the reason

## Step 4 (Optional): Add Authentication

By default Intaris accepts all requests. To protect the server:

```bash
# Single shared key
export INTARIS_API_KEY=your-secret-key

# Or multi-key with user mapping (recommended)
export INTARIS_API_KEYS='{"key-for-alice": "alice@example.com", "key-for-bob": "bob@example.com"}'
```

Clients send the key via `Authorization: Bearer <key>` or `X-API-Key: <key>` header. When using a single shared key, clients must also send `X-User-Id` to identify themselves.

See the [Configuration Reference](configuration.md) for all environment variables.

## Step 5 (Optional): Conversation Search

Intaris ships a **Search** tab in the management UI that lets you find
sessions by content across three kinds: summaries, intentions, and
reasoning. The lexical tier (Postgres `tsvector` + GIN, or plain
`LIKE` on SQLite) is enabled by default — no extra configuration
needed. It reads directly from the canonical tables you already have.

For semantic / multilingual recall the optional vector tier adds
either pgvector (dense embeddings, Postgres-only) or **Qdrant native
dense + sparse hybrid** (works with any backend).

### What you get with each mode

| Capability | Default (lexical only) | + Vector tier enabled |
|---|---|---|
| Token / phrase match | Yes | Yes |
| Diacritic-insensitive match | Yes | Yes |
| Typo / fuzzy / partial-word | Limited (PG `pg_trgm` if installed) | Yes |
| Synonym / paraphrase recall | No | Yes |
| Multilingual (query EN, content CS, ...) | No | Yes (with a multilingual embedding model) |
| Required infrastructure | None — uses your existing DB | Postgres + `pgvector`, OR Qdrant (server URL or local-mode path) |
| Embedding API key | Not needed | Required |
| Per-write cost | Zero (PG generated columns) | One embedding call per audit / summary write |
| Per-query cost | Single SQL query | Embedding call + vector search + fusion |
| Storage overhead | Small (`tsvector`) or none (SQLite) | Dense vectors at `dim x 4` bytes per row |
| Backfill on enable | Not needed | One-time walk of audit_log + summaries |

### Provider comparison

| Aspect | `disabled` (default) | `pgvector` | `qdrant` |
|---|---|---|---|
| DB requirement | Any (PG or SQLite) | **Postgres only** | Any (PG or SQLite) |
| Extra service | None | None — runs inside Postgres | Qdrant when URL-mode; **none** in local-mode |
| Vector flavor | n/a | Dense only | **Dense + sparse** native hybrid |
| Token recall | Lexical only | PG lexical RRF-fused with vector | Qdrant sparse BM25 server-side fused with dense |
| Best for | Cheap defaults, single-user quickstart | Postgres deployments wanting semantic recall without a new service | Multilingual / paraphrase-heavy content; serverless single-user (local-mode) or multi-tenant (server URL) |
| Install | built in | built in (needs `vector` extension on PG) | built in |

### Single-user / quickstart with serverless Qdrant

The simplest production-quality setup pairs Intaris on SQLite with
Qdrant in **local-mode** (an embedded, SQLite-backed Qdrant —
no service required). Combined with a local Ollama embedding server,
the entire stack runs without any external services:

```bash
pip install intaris
ollama pull bge-m3

export INTARIS_SEARCH_VECTOR_PROVIDER=qdrant
export INTARIS_SEARCH_QDRANT_URL=~/.intaris/qdrant         # local-mode path
export INTARIS_SEARCH_EMBEDDING_MODEL=bge-m3
export INTARIS_SEARCH_EMBEDDING_DIM=1024
export INTARIS_SEARCH_EMBEDDING_BASE_URL=http://localhost:11434/v1
export INTARIS_SEARCH_EMBEDDING_API_KEY=ollama
```

When the path looks like a filesystem path (`~/.intaris/qdrant`,
`/abs/path`, or `file:///abs/path`) Intaris automatically uses Qdrant
local-mode — there is no Qdrant service to run, and the index
persists alongside your SQLite database.

### Postgres deployments

If Intaris already runs on Postgres, pgvector is the lowest-overhead
vector option (no extra service):

```bash
export INTARIS_SEARCH_VECTOR_PROVIDER=pgvector
export INTARIS_SEARCH_EMBEDDING_MODEL=text-embedding-3-small
export INTARIS_SEARCH_EMBEDDING_DIM=1536
export INTARIS_SEARCH_EMBEDDING_API_KEY=$OPENAI_API_KEY
```

For shared deployments with an existing Qdrant cluster, point
`INTARIS_SEARCH_QDRANT_URL` at the cluster URL.

To turn search off entirely set `INTARIS_SEARCH_ENABLED=false`. See
the [Conversation Search section in AGENTS.md](../AGENTS.md#conversation-search)
for the full architecture.

## What's Next

- [Architecture](architecture.md) — Understand how Intaris evaluates tool calls
- [Evaluation Pipeline](evaluation-pipeline.md) — Classification, LLM evaluation, and decision matrix
- [Configuration](configuration.md) — Tune LLM models, timeouts, and rate limits
- [Management UI](management-ui.md) — Monitor sessions and approve escalations
- [MCP Proxy](mcp-proxy.md) — Proxy upstream MCP servers through Intaris
- [Deployment](deployment.md) — Production deployment with Docker and authentication
