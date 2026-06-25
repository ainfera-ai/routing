-- AIN-335 / AIN-550 · routing_outcomes -> JSON dump for the LinUCB training cadence.
--
-- READ-ONLY. The "thin, documented query step" the cadence orchestrator expects upstream
-- (training_cadence.py / .sh consume the resulting JSON via export_outcomes.project_rows).
-- Parameterized by :days (e.g. `psql "$DATABASE_URL" -v days=30 -f scripts/dump_routing_outcomes.sql`).
--
-- Emits a single JSON array of the columns export_outcomes consumes. tenant_id is cast to
-- text so the model-free cell (task:tenant:band) matches the api consumer + refit byte-for-byte;
-- created_at is ISO-8601 UTC for deterministic tick ordering.
SELECT coalesce(json_agg(row_to_json(t)), '[]'::json)
FROM (
  SELECT
    task_type,
    cell,
    chosen_model_slug,
    reward,
    -- AIN-621 · reward PROVENANCE: export_outcomes authority-allowlists the cost corpus to
    -- reward_source IN ('council','verify'). Without this column the allowlist drops every
    -- row and project_rows raises INVARIANT 2 (fail-loud, never a silent empty corpus).
    reward_source,
    policy_version,
    to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
    judge_status,
    source,
    tenant_id::text AS tenant_id,
    fleet_agent,
    traffic_origin,
    traffic_class
  FROM routing_outcomes
  WHERE reward IS NOT NULL
    AND created_at >= now() - ((:days)::int * interval '1 day')
) t;
