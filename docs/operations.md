# Resilient agent operation

The runtime restores its last SQLite/WAL world snapshot before it admits live
autonomy. It then replays the non-consuming 100-message inbox, reads both
server action queues, and queues one comprehensive command-center status
report. An already queued status report is detected and not duplicated. A
live Socket.IO connection performs another gap-fill after authentication to
close the small window between startup reconciliation and connection.

Persistence, action reconciliation, socket transport, and the planning
pipeliner are independently supervised. An unexpected exception or clean
exit restarts that service with exponential backoff (0.25 seconds initially,
5 seconds maximum by default). A deliberate shutdown disables restarts before
the services are stopped and the final snapshot is flushed.

## systemd installation

The supplied unit assumes the repository and virtual environment are installed
under `/opt/drone-ctf` and runs them as a dedicated `drone-agent` user.

```bash
sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin drone-agent
sudo cp deploy/drone-agent.service /etc/systemd/system/drone-agent.service
sudo cp deploy/drone-agent.env.example /etc/drone-agent.env
sudo chmod 0600 /etc/drone-agent.env
sudoedit /etc/drone-agent.env
sudo systemctl daemon-reload
sudo systemctl enable --now drone-agent
```

The unit restarts a failed or killed process after two seconds. systemd creates
and grants access to `/var/lib/drone-agent` for snapshots and
`/var/log/drone-agent` for replay telemetry; the rest of the host filesystem is
read-only to the service.

## Restart acceptance check

Run this against the private server while autonomy is active:

```bash
since=$(date --iso-8601=ns)
started_at=$(date +%s)
sudo systemctl kill --signal=KILL --kill-whom=main drone-agent
until systemctl is-active --quiet drone-agent && \
  journalctl -u drone-agent --since="$since" --no-pager | \
  grep -q "Transport ready"; do
    test $(( $(date +%s) - started_at )) -lt 30 || exit 1
    sleep 1
done
```

Then inspect the same journal window for `Restored world snapshot` and
`Startup resync`. The service should return to `Transport ready` in under 30
seconds, and the restored snapshot cycle should precede the gap-filled status
updates. With the default cadence, an ungraceful kill loses at most about 10
seconds of purely local world changes; server messages and queues repair that
window during resync.
