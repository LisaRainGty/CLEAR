# Status — sources_only fair rerun (2026-07-21)

## Done

- [x] Fair protocol: `evidence_policy=sources_only` everywhere
- [x] `main/` reproducibility package assembled
- [x] Pushed to https://github.com/LisaRainGty/CLEAR (`main` branch)
- [x] Remote GPU `hz-t3.matpool.com:26637` (A30 24GB) deployed at `/root/CLEAR`
- [x] Full paper run started in tmux `clear_paper` (`scripts/run_paper_all.sh`)

## Monitor remote

```bash
ssh -p 26637 root@183.136.238.107   # or: ssh clear-gpu
tmux a -t clear_paper
tail -f /root/CLEAR/results/run_paper_all.log
wc -l /root/CLEAR/results/campaign6_results.jsonl
```

## After completion

```bash
rsync -azP -e 'ssh -i ~/.ssh/claimarc_matpool -p 26637' \
  root@183.136.238.107:/root/CLEAR/results/ \
  "./results/"
```

Then update paper tables from the new jsonl aggregates.
