"""
durability_declarations.py — authoritative durability classifications for
Hermes tools, keyed by tool name (the name used in registry.register).

Imported at tool discovery time. Every live tool MUST have an entry here
before `durable_runtime: true` can be enabled on any profile — preflight
enforces this.

This file is the human-reviewed, machine-enforced source of truth. Changes
REQUIRE:
  - entry in docs/specs/tool-classification-manifest.yaml (needs_review: false)
  - two reviewer names in the manifest entry for @durable_write classifications
  - three reviewers for @durable_stateful

See docs/specs/TOOL_CLASSIFICATION_REVIEW.md for the review process.
"""

from __future__ import annotations

# Import path is from hermes-agent root; works whether loaded by the
# gateway or by a test harness that sys.path-inserts hermes-agent.
from agent.idempotency import register_durability, register_mcp_durability


# ─────────────────────────────────────────────────────────────────────────────
# Wave 1 — confirmed reads (2026-04-22)
#
# Reviewed by: deacon
# Status: 11 reads + 1 retroactive add (ha_list_services — classifier tagged
# write due to sibling function signals, source inspection confirms read).
# Deferred to later waves:
#   - clarify (write/degraded — user-facing prompt)
#   - cronjob (write/queryable — creates/updates/deletes cron jobs)
#   - delegate_task (write/at_most_once — spawns subagent subprocesses)
#   - skill_manage (write/at_most_once — creates/updates/deletes skill files)
#   - browser_console (write/at_most_once — runs arbitrary JS via eval)
#   - browser_vision (write/at_most_once — writes screenshot files)
#   - browser_get_images (write/at_most_once — uses JS eval mechanism)
#   - web_search, web_extract (deferred pending lambda-handler review)
# ─────────────────────────────────────────────────────────────────────────────

register_durability("analyze_video_frames",  class_="read")
register_durability("browser_snapshot",      class_="read")
register_durability("ha_get_state",          class_="read")
register_durability("ha_list_entities",      class_="read")
register_durability("ha_list_services",      class_="read")  # retro-added
register_durability("mixture_of_agents",     class_="read")
register_durability("read_file",             class_="read")
register_durability("search_files",          class_="read")
register_durability("session_search",        class_="read")
register_durability("skill_view",            class_="read")
register_durability("skills_list",           class_="read")
register_durability("vision_analyze",        class_="read")


# ─────────────────────────────────────────────────────────────────────────────
# Wave 2 — at_most_once writes (2026-04-22)
#
# Reviewers: deacon, claude-opus-4.7
# Status: 18 writes verified as at_most_once — no native idempotency primitive,
# duplicate-effect risk on resume is worse than missed-effect risk, so the
# graph will NEVER auto-reissue these on resume. Operator manually replays
# via DLQ if needed.
# ─────────────────────────────────────────────────────────────────────────────

# File / patch / exec
register_durability("write_file",       class_="write", reconciliation="at_most_once")
register_durability("patch",            class_="write", reconciliation="at_most_once")
register_durability("execute_code",     class_="write", reconciliation="at_most_once")
register_durability("terminal",         class_="write", reconciliation="at_most_once")
register_durability("process",          class_="write", reconciliation="at_most_once")

# Business records and optional outbound receipt delivery
register_durability("business_receipt", class_="write", reconciliation="at_most_once")

# Subagent spawning
register_durability("delegate_task",    class_="write", reconciliation="at_most_once")

# Skill management (file CRUD in ~/.hermes/skills/)
register_durability("skill_manage",     class_="write", reconciliation="at_most_once")

# Home Assistant service invocations (on/off, press, etc. — not all idempotent)
register_durability("ha_call_service",  class_="write", reconciliation="at_most_once")

# Image generation (file artifact)
register_durability("image_generate",   class_="write", reconciliation="at_most_once")

# Browser DOM mutations
register_durability("browser_navigate", class_="write", reconciliation="at_most_once")
register_durability("browser_click",    class_="write", reconciliation="at_most_once")
register_durability("browser_type",     class_="write", reconciliation="at_most_once")
register_durability("browser_scroll",   class_="write", reconciliation="at_most_once")
register_durability("browser_back",     class_="write", reconciliation="at_most_once")
register_durability("browser_press",    class_="write", reconciliation="at_most_once")

# Browser capture operations that are actually writes (JS eval / file writes)
register_durability("browser_console",    class_="write", reconciliation="at_most_once")
register_durability("browser_vision",     class_="write", reconciliation="at_most_once")
register_durability("browser_get_images", class_="write", reconciliation="at_most_once")


# ═════════════════════════════════════════════════════════════════════════════
# MCP Wave 1 — classifications for Enoch's enabled MCP servers (2026-04-22)
#
# 77 tools discovered across 9 enabled local MCP servers:
#   doc-monitor, four-pillars, local-memory, local-observability,
#   local-resources, local-vault, openrouter-image, storm-watcher,
#   telegram-admin
#
# Reviewer: deacon + claude-opus-4.7 (source inspection)
# Strategy:
#   - 31 read tools explicitly registered (source-confirmed pure reads).
#   - 1 write override (store_fact — auto-classifier missed "store" verb).
#   - Remaining ~45 write tools: NOT explicitly registered. They hit
#     MCP_RUNTIME_DEFAULT (write/at_most_once) — safe conservative fallback.
#     Operator reviews those individually before graduating to queryable.
# ═════════════════════════════════════════════════════════════════════════════

# doc-monitor reads
register_mcp_durability("doc-monitor", "doc_friction_scan",  class_="read")
register_mcp_durability("doc-monitor", "doc_health_detail",  class_="read")
register_mcp_durability("doc-monitor", "doc_pulse",          class_="read")

# four-pillars reads (pure-function scorecard queries)
register_mcp_durability("four-pillars", "four_pillars_cap_audit",               class_="read")
register_mcp_durability("four-pillars", "four_pillars_dossier_pdf",             class_="read")
register_mcp_durability("four-pillars", "four_pillars_evidence_sources_status", class_="read")
register_mcp_durability("four-pillars", "four_pillars_find_similar",            class_="read")
register_mcp_durability("four-pillars", "four_pillars_label_query",             class_="read")
register_mcp_durability("four-pillars", "four_pillars_lookup_evidence",         class_="read")
register_mcp_durability("four-pillars", "four_pillars_profile",                 class_="read")
register_mcp_durability("four-pillars", "four_pillars_search_scorecard",        class_="read")
register_mcp_durability("four-pillars", "four_pillars_status",                  class_="read")

# local-memory reads + one write
register_mcp_durability("local-memory", "memory_search",     class_="read")
register_mcp_durability("local-memory", "recent_context",    class_="read")
register_mcp_durability("local-memory", "transcript_search", class_="read")
# store_fact auto-classifier misclassified as read; source = INSERT into memories.
# Queryable because row has stable id.
register_mcp_durability("local-memory", "store_fact",
                        class_="write", reconciliation="queryable")

# local-observability reads
register_mcp_durability("local-observability", "gateway_health", class_="read")
register_mcp_durability("local-observability", "recent_errors",  class_="read")

# local-resources reads
register_mcp_durability("local-resources", "resource_get",     class_="read")
register_mcp_durability("local-resources", "resource_list",    class_="read")
register_mcp_durability("local-resources", "template_render",  class_="read")

# local-vault reads
register_mcp_durability("local-vault", "vault_list",   class_="read")
register_mcp_durability("local-vault", "vault_read",   class_="read")
register_mcp_durability("local-vault", "vault_search", class_="read")

# openrouter-image reads
register_mcp_durability("openrouter-image", "healthcheck",       class_="read")
register_mcp_durability("openrouter-image", "list_image_models", class_="read")

# storm-watcher reads
register_mcp_durability("storm-watcher", "check_storm_watch",    class_="read")
register_mcp_durability("storm-watcher", "get_active_alerts",    class_="read")
register_mcp_durability("storm-watcher", "get_hail_lead_zones",  class_="read")
register_mcp_durability("storm-watcher", "get_storm_reports",    class_="read")

# telegram-admin reads
register_mcp_durability("telegram-admin", "telegram_admin_channel_lookup", class_="read")
register_mcp_durability("telegram-admin", "telegram_admin_read_history",   class_="read")
