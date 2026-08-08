# Assets

Not tracked in git — large binaries and personal recordings. Keep backups.

| Path | What | Required |
|---|---|---|
| `voice/reference.wav` | 60–120 s of you reading calmly, clean audio | yes |
| `avatar/base_loop.mp4` | 3–5 min of you on camera, listening not speaking | yes (unless `renderer: none`) |
| `brand/background.png` | 1920×1080 backdrop | yes |
| `brand/intro.mp4` | Opening bumper | optional |
| `brand/outro.mp4` | Closing bumper | optional |
| `brand/bed.mp3` | Music bed, sidechain-ducked under speech | optional |
| `broll/*.png,jpg,mp4` | Cutaway imagery, named descriptively | optional |

B-roll is matched to a script's visual cues by filename word overlap, so name
files the way you would describe them: `town-council-chamber.jpg`,
`post6-hall-exterior.jpg`, `franklin-street-traffic.jpg`.

See [docs/RUNBOOK.md](../docs/RUNBOOK.md) for how to record the first two.
