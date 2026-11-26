# lerobot runner on supertower

Adds a second GitHub Actions runner service on the existing `supertower` host, registered only to `kinetic-auto/lerobot`.

It does not install Docker, change the hostname, or touch the manager runner at `/home/kinetic/actions-runner`.

| | |
|---|---|
| Directory | `/home/kinetic/actions-runner-lerobot` |
| Service | `actions.runner.kinetic-auto-lerobot.supertower.service` |
| Alias | `lerobot-runner.service` |
| Labels | `self-hosted`, `Linux`, `X64`, `supertower` |
| SSH | `kinetic@192.168.1.219` |

`gh auth` on this machine must be able to admin `kinetic-auto/lerobot`. The registration token is minted here and is not written to the host.

```bash
cd .github/setup
./runner-configure.sh
```

The first run creates `.github/setup/venv/` and installs the Python dependencies from `requirements.txt`.
