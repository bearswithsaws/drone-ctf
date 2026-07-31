# `@drone-ctf/commander-contract`

Versioned TypeScript types for the commander HTTP API and Socket.IO stream.
Package major version `1` carries `contract_version: "1.0"` in state snapshots
and world diffs.

```ts
import {
  CONTRACT_VERSION,
  type Directive,
  type StateSnapshot,
  type SocketEvent,
} from "@drone-ctf/commander-contract";
```

The public package is hand-written in `src/index.ts`. Python runtime DTOs live
in `agent/commander/contract.py`; their generated JSON Schema is exported as
`@drone-ctf/commander-contract/schema`. `npm run check` verifies the checked-in
schema, generated compatibility types, hand-written types, and publishable
build all agree.
