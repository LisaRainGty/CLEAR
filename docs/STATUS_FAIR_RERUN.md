# Status — sources_only fair rerun (2026-07-21)

## Done locally

- [x] Converted experiment chapter to `docs/实验结果与分析.md`
- [x] Forced paper-fair evidence policy: **sources_only** (params/OCR/VLM only)
- [x] Updated `train` / `data` / `baselines` / campaigns 6–8 / geom / xdom
- [x] Built clean `main/` package (code + supervised/all jsonl + scripts + docs)
- [x] Local git commits in `main/` ready for `github.com/LisaRainGty/CLEAR`

## Blocked by network (needs you)

### 1) Matpool GPU SSH

Outbound SSH is being reset before auth (`Connection closed by … port 29378`).
System DNS maps `px-asia-3.matpool.com` → Clash fake-ip `198.18.0.127`.
Real A record is `211.72.37.226`, but SSH handshake still drops (likely **Clash TUN /
system proxy intercepting SSH**).

**Please:**
1. Confirm the Matpool instance is **running** and the mapped port is still **29378**.
2. Temporarily disable Clash TUN / system proxy (or add DIRECT rule for `211.72.37.226` and GitHub SSH).
3. Test: `ssh -p 29378 root@211.72.37.226`
4. Then from `main/`:  
   `REMOTE=root@211.72.37.226 REMOTE_PORT=29378 bash scripts/deploy_remote.sh`

**Security:** the Matpool password was pasted in chat — rotate it on the Matpool console after use. Never commit passwords.

### 2) GitHub push to CLEAR

Local commits exist, but `git push` failed: no HTTPS credentials and SSH port 22 is also blocked by the same proxy.

**Please (pick one):**
```bash
cd "/Volumes/My Passport/claimarc_final/main"
# Option A — HTTPS with PAT
git remote set-url origin https://<TOKEN>@github.com/LisaRainGty/CLEAR.git
git push -u origin main

# Option B — fix SSH proxy, then:
git remote set-url origin git@github.com:LisaRainGty/CLEAR.git
git push -u origin main
```

## After remote runs finish

```bash
rsync -azP -e 'ssh -p 29378' root@HOST:/root/CLEAR/results/ ./results/
# Update paper tables from new jsonl aggregates; keep legacy/ as reference only.
```
