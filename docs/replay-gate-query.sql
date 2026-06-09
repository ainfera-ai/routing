-- AIN-335 replay-gate · the read-only query that produces the input bundle's
-- `aggregates` block. Kept here (not in the dependency-light library) so the
-- bundle is reproducible. Run against the prod routing_outcomes store; combine
-- with the enrolled-catalog snapshot + the api CELL_MIN_QUALITY floor map to
-- form the bundle replay_gate.py consumes.
--
-- Per-(bandit_cell, arm, temporal-split, outcome_class): row count + reward sum.
-- Temporal split = per-cell 70th-percentile created_at (train < cutoff <= holdout).
WITH base AS (
  SELECT
    CASE WHEN array_length(string_to_array(cell, ':'), 1) = 3
         THEN split_part(cell, ':', 1) || ':' || split_part(cell, ':', 3)
         ELSE coalesce(task_type, 'general') || ':balanced' END AS bandit_cell,
    split_part(cell, ':', 1) AS task,
    chosen_model_slug AS arm,
    reward::float8 AS reward,
    CASE WHEN outcome_status = 'succeeded' THEN 'succeeded'
         WHEN outcome_status LIKE 'failed%' THEN 'failed'
         ELSE 'other' END AS outcome_class,
    created_at
  FROM routing_outcomes
  WHERE source = 'prod'
    AND chosen_model_slug IS NOT NULL
    AND reward IS NOT NULL
    AND (judge_status IS NULL OR judge_status = 'labeled')
    AND lower(coalesce(traffic_origin, '')) NOT IN ('degraded', 'mlx', 'mlx-degraded')
),
cut AS (
  SELECT bandit_cell, percentile_disc(0.7) WITHIN GROUP (ORDER BY created_at) AS cutoff
  FROM base GROUP BY bandit_cell
),
split AS (
  SELECT b.*, CASE WHEN b.created_at < c.cutoff THEN 'train' ELSE 'holdout' END AS split
  FROM base b JOIN cut c USING (bandit_cell)
)
SELECT bandit_cell, task, arm, split, outcome_class,
       count(*) AS n, round(sum(reward)::numeric, 4) AS reward_sum
FROM split
GROUP BY 1, 2, 3, 4, 5
ORDER BY bandit_cell, arm, split, outcome_class;
