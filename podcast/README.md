# Private podcast feed

Replaces the Tailscale link with per-listener feed URLs that work in any podcast
app. Each of the ten listeners gets their own URL. Revoking one person is a
single command and does not disturb anyone else.

```
NAS  \\MMRUNRAIDNAS\podcasts
  |  publish.mjs uploads new files
  v
Cloudflare R2  (audio + episodes.json)
  ^
  |  Worker checks the token, then streams
Listener  https://<your-domain>/f/<token>.xml
```

## What each listener sees

They get one link. Opening it in a browser shows a short page with their feed
URL and setup steps for their app; pasting it into a podcast app subscribes them
and every new episode arrives on its own.

**Apple Podcasts on iPhone cannot add a feed by URL** — Apple removed that. The
onboarding page points iPhone listeners at Overcast or Pocket Casts, both free.
The Mac Podcasts app *can* add one, via **File → Add a Show by URL**. This is a
limitation of Apple's app, not of this setup; the only way into iOS Apple
Podcasts is a fully public, Apple-listed show.

The feed carries `<itunes:block>Yes</itunes:block>`, which tells Apple and other
directories not to index it even if a URL leaks.

## One-time setup

You need a free Cloudflare account. Run everything from this `podcast/` folder.

**1. Install and sign in**

```bash
npm install -g wrangler
wrangler login
```

**2. Create the storage**

```bash
wrangler r2 bucket create post6-podcast
wrangler kv namespace create SUBSCRIBERS
```

The second command prints a namespace `id`. Paste it into `worker/wrangler.toml`
in place of `REPLACE_WITH_KV_NAMESPACE_ID`.

**3. Configure**

```bash
cp podcast.config.example.json podcast.config.json
```

Edit it. `sourceDir` is already set to `\\MMRUNRAIDNAS\podcasts`. If Node has
trouble with the UNC path, map the share to a drive letter in Windows and use
`"P:\\"` instead. Leave `feedOrigin` alone for now.

**4. Deploy**

```bash
cd worker && wrangler deploy && cd ..
```

Wrangler prints the live URL, something like
`https://post6-podcast.yourname.workers.dev`. Put that in `feedOrigin` in
`podcast.config.json`.

To use `feed.alpost6.org` instead, uncomment the `[[routes]]` block in
`worker/wrangler.toml`, redeploy, and update `feedOrigin` to match. The domain
has to be on Cloudflare for this to work.

**5. Publish your back catalogue**

```bash
node scripts/publish.mjs --dry-run   # check what it found
node scripts/publish.mjs
```

**6. Add your listeners**

```bash
node scripts/subscribers.mjs add "Dave K" dave@example.com
```

That prints the link to send them. Repeat for all ten.

## Daily routine

Drop the episode on the NAS exactly as you do today, then:

```bash
node scripts/publish.mjs
```

Already-published files are skipped, so it is safe to run any time. Episode
titles come from the filename — `2026-08-09-morning-report.mp3` becomes
"Morning Report" dated 9 August. To write your own titles or add show notes,
edit `local/episodes.json` after publishing and re-upload it:

```bash
wrangler r2 object put post6-podcast/episodes.json --file=local/episodes.json --content-type=application/json --remote
```

If `ffmpeg` is installed, `ffprobe` supplies episode durations automatically. It
is optional; without it the duration tag is simply left out.

## Managing listeners

```bash
node scripts/subscribers.mjs list                # who has access, and when they last checked in
node scripts/subscribers.mjs add "Name" [email]  # mint a new link
node scripts/subscribers.mjs revoke "Name"       # kill one link
```

`list` shows a per-person check-in count, so a link being shared around tends to
show up as one listener with far more activity than the rest. Revocation takes
about a minute to reach every Cloudflare edge location.

## What this costs

Nothing, at this size. R2 gives 10 GB of storage free and charges **no egress
fees**; a 30-minute episode at 64 kbps mono is roughly 14 MB, so a year of daily
episodes is about 5 GB. Workers' free tier is 100,000 requests per day and ten
subscribers polling hourly use a few hundred. KV free tier covers the token
lookups with room to spare.

Watch the R2 free tier at around the two-year mark. Deleting old audio from the
bucket and its entry from `episodes.json` is the easy remedy.

## Security model, plainly

Access rests on the token in the URL being unguessable — 16 random bytes, which
is not brute-forceable at any rate a Worker will serve. It is not a login, and
it cannot stop a determined listener from forwarding their link to a friend.
What it does give you is **attribution and individual revocation**: every link
is distinct, so a leak is traceable to one person and closed with one command.

That is the right trade for a ten-person daily show. A real login would mean
Cloudflare Access in front of a web player, which no podcast app can get
through — listeners would have to visit a webpage every day instead, and a daily
show does not survive that.

**Nothing secret lives in this repository.** Tokens exist only in Cloudflare KV;
`podcast.config.json` and `local/` are gitignored. The Worker source being
public costs you nothing.

## Alternative worth knowing

Your NAS already runs Tailscale, so it could instead run `cloudflared` and serve
the audio through a Cloudflare Tunnel, skipping the upload step and the storage
limit entirely. The cost is that the NAS must be online and reachable whenever
anyone listens, and your home upstream carries the traffic. Copying to R2 keeps
the NAS off the public internet and keeps working when it is down — which is why
it is the default here.
