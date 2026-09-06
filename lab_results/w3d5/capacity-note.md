# Capacity note (team, one page)

## The numbers

* Locked model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
* Target p95 end-to-end latency (your SLO today): `2.0` seconds
* Knee concurrency (highest concurrency whose p95 is still under target): `16`
* Tokens per second at the knee: `1107.77`
* Max sustainable request rate at the target p95: `8.31 req/s`

## The limiting family

One sentence, using this morning's triage lens (compute vs memory vs overhead): which family limits this stack at the knee, and the tell that points to it.

* Compute-bound: throughput continues to increase with concurrency, but latency rises from 0.98s at concurrency 1 to 1.51s at concurrency 16, indicating the model is approaching available compute capacity rather than being limited by fixed overhead.

## Why the knee, not the peak

One sentence in your own words on why you report the knee at the SLO rather than the peak throughput.

* The knee is reported because it represents the highest throughput that still meets the latency SLO, while peak throughput may deliver more work but at a response time that no longer satisfies user-facing requirements.
