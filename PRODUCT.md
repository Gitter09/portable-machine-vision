# Portable vision dashboard

## Platform
Web; offline-capable local dashboard for a large 1080p or 4K summit monitor.

## Users and purpose
Summit visitors, mostly salespeople and occasionally operators. Story approved by owner: “See what’s missing. Right at the edge.” Demonstrate IC and screw-terminal presence/absence detection, not product pass/fail judgement.

## Constraints
White industrial interface, accessible contrast, simple colours, consistent downloaded icons. Supplied BITS Pilani WILP logo. Footage currently unavailable; illustrative simulation must be explicitly labelled. Confidence is not accuracy; repeated frames are not units inspected. No accuracy, yield, savings or optical-latency claims. Existing receiver pairs coordinates with exact source-frame hashes; stale and unmatched results must clear overlays. Board sessions remain bounded to 300 seconds; no hardware changes or automatic restart.

## Implementation
Extend the repository’s plain HTML/CSS/JavaScript browser approach with a standalone local Python server and read-only loopback receiver bridge. No changes to copied SDK or frozen deployment payloads.
