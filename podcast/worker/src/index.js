/**
 * Private podcast feed worker.
 *
 * Routes:
 *   GET  /f/<token>          -> onboarding page (how to add the feed to an app)
 *   GET  /f/<token>.xml      -> RSS feed for that subscriber
 *   GET|HEAD /a/<token>/<key> -> audio / cover art from R2, with Range support
 *
 * Anything else, or an unknown token, is a 404. There is deliberately no
 * "invalid token" message: an outsider probing the domain cannot tell a
 * revoked token from a wrong one.
 */

const SHOW_KEY = 'episodes.json';
const TOUCH_INTERVAL_MS = 60 * 60 * 1000; // rate-limit the KV write to hourly

export default {
  async fetch(request, env, ctx) {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return notFound();
    }

    const url = new URL(request.url);
    const parts = url.pathname.split('/').filter(Boolean);

    if (parts.length === 2 && parts[0] === 'f') {
      const raw = parts[1];
      const isFeed = raw.endsWith('.xml');
      const token = isFeed ? raw.slice(0, -4) : raw;

      const subscriber = await lookup(env, token);
      if (!subscriber) return notFound();
      ctx.waitUntil(touch(env, token, subscriber));

      return isFeed
        ? buildFeed(env, url, token)
        : onboardingPage(url, token, subscriber);
    }

    if (parts.length >= 3 && parts[0] === 'a') {
      const token = parts[1];
      const key = decodeURIComponent(parts.slice(2).join('/'));

      const subscriber = await lookup(env, token);
      if (!subscriber) return notFound();
      ctx.waitUntil(touch(env, token, subscriber));

      return serveObject(request, env, key);
    }

    return notFound();
  },
};

/* ------------------------------------------------------------------ auth */

async function lookup(env, token) {
  // Tokens are opaque and high-entropy, so a plain KV lookup by key is the
  // whole check. A missing key means "never existed" or "revoked" alike.
  if (!token || token.length < 16 || !/^[A-Za-z0-9_-]+$/.test(token)) return null;
  return env.SUBSCRIBERS.get(`sub:${token}`, { type: 'json' });
}

async function touch(env, token, subscriber) {
  const now = Date.now();
  const last = Date.parse(subscriber.lastSeen || '') || 0;
  if (now - last < TOUCH_INTERVAL_MS) return;

  const updated = {
    ...subscriber,
    lastSeen: new Date(now).toISOString(),
    hits: (subscriber.hits || 0) + 1,
  };
  await env.SUBSCRIBERS.put(`sub:${token}`, JSON.stringify(updated));
}

/* ------------------------------------------------------------------ feed */

async function buildFeed(env, url, token) {
  const object = await env.MEDIA.get(SHOW_KEY);
  if (!object) return notFound();

  let data;
  try {
    data = await object.json();
  } catch {
    return new Response('Feed data is not valid JSON', { status: 500 });
  }

  const show = data.show || {};
  const base = `${url.origin}/a/${token}`;
  const episodes = [...(data.episodes || [])].sort(
    (a, b) => Date.parse(b.pubDate || 0) - Date.parse(a.pubDate || 0),
  );

  const items = episodes.map((ep) => {
    const enclosureUrl = `${base}/${encodePath(ep.file)}`;
    return `    <item>
      <title>${esc(ep.title)}</title>
      <guid isPermaLink="false">${esc(ep.id || ep.file)}</guid>
      <pubDate>${rfc2822(ep.pubDate)}</pubDate>
      <description>${esc(ep.description || '')}</description>
      <itunes:summary>${esc(ep.description || '')}</itunes:summary>
      <enclosure url="${esc(enclosureUrl)}" length="${Number(ep.size) || 0}" type="${esc(ep.type || 'audio/mpeg')}"/>
${ep.duration ? `      <itunes:duration>${esc(ep.duration)}</itunes:duration>\n` : ''}      <itunes:episodeType>full</itunes:episodeType>
    </item>`;
  });

  const coverUrl = show.coverArt ? `${base}/${encodePath(show.coverArt)}` : '';

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>${esc(show.title || 'Private Podcast')}</title>
    <link>${esc(show.link || url.origin)}</link>
    <description>${esc(show.description || '')}</description>
    <language>${esc(show.language || 'en-us')}</language>
    <itunes:author>${esc(show.author || '')}</itunes:author>
    <itunes:summary>${esc(show.description || '')}</itunes:summary>
    <itunes:explicit>false</itunes:explicit>
    <itunes:type>episodic</itunes:type>
    <!-- Keeps the show out of Apple's and other directories' indexes. -->
    <itunes:block>Yes</itunes:block>
    <itunes:complete>No</itunes:complete>
${show.ownerEmail ? `    <itunes:owner>\n      <itunes:name>${esc(show.author || '')}</itunes:name>\n      <itunes:email>${esc(show.ownerEmail)}</itunes:email>\n    </itunes:owner>\n` : ''}${coverUrl ? `    <itunes:image href="${esc(coverUrl)}"/>\n` : ''}${items.join('\n')}
  </channel>
</rss>
`;

  return new Response(xml, {
    headers: {
      'Content-Type': 'application/rss+xml; charset=utf-8',
      // Private: never let an intermediary cache a tokenised feed.
      'Cache-Control': 'private, no-store',
      'X-Robots-Tag': 'noindex, nofollow',
    },
  });
}

/* ----------------------------------------------------------------- media */

async function serveObject(request, env, key) {
  if (key === SHOW_KEY) return notFound(); // metadata is served as RSS only

  const head = await env.MEDIA.head(key);
  if (!head) return notFound();

  const headers = new Headers();
  head.writeHttpMetadata(headers);
  headers.set('ETag', head.httpEtag);
  headers.set('Accept-Ranges', 'bytes');
  headers.set('Cache-Control', 'private, max-age=3600');
  headers.set('X-Robots-Tag', 'noindex, nofollow');

  if (request.method === 'HEAD') {
    headers.set('Content-Length', String(head.size));
    return new Response(null, { status: 200, headers });
  }

  const rangeHeader = request.headers.get('Range');
  if (!rangeHeader) {
    const object = await env.MEDIA.get(key);
    if (!object) return notFound();
    headers.set('Content-Length', String(head.size));
    return new Response(object.body, { status: 200, headers });
  }

  const range = parseRange(rangeHeader, head.size);
  if (!range) {
    headers.set('Content-Range', `bytes */${head.size}`);
    return new Response(null, { status: 416, headers });
  }

  const length = range.end - range.start + 1;
  const object = await env.MEDIA.get(key, {
    range: { offset: range.start, length },
  });
  if (!object) return notFound();

  headers.set('Content-Range', `bytes ${range.start}-${range.end}/${head.size}`);
  headers.set('Content-Length', String(length));
  return new Response(object.body, { status: 206, headers });
}

function parseRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match) return null;

  const [, startRaw, endRaw] = match;
  if (startRaw === '' && endRaw === '') return null;

  let start;
  let end;
  if (startRaw === '') {
    const suffix = Number.parseInt(endRaw, 10);
    if (!Number.isFinite(suffix) || suffix <= 0) return null;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number.parseInt(startRaw, 10);
    end = endRaw === '' ? size - 1 : Number.parseInt(endRaw, 10);
  }

  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start >= size || start > end) return null;
  return { start, end: Math.min(end, size - 1) };
}

/* ------------------------------------------------------------ onboarding */

function onboardingPage(url, token, subscriber) {
  const feedUrl = `${url.origin}/f/${token}.xml`;
  const name = subscriber.name ? esc(subscriber.name) : 'there';

  const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Your private podcast feed</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 34rem; margin: 0 auto; padding: 2rem 1.25rem; line-height: 1.55; }
  h1 { font-size: 1.4rem; margin-bottom: .25rem; }
  p.lede { margin-top: 0; opacity: .75; }
  .feed { display: flex; gap: .5rem; margin: 1.5rem 0; }
  input { flex: 1; padding: .6rem .7rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: .8rem; border: 1px solid currentColor; border-radius: .4rem; background: transparent;
          color: inherit; min-width: 0; }
  button { padding: .6rem 1rem; border-radius: .4rem; border: 1px solid currentColor;
           background: transparent; color: inherit; cursor: pointer; font-size: .85rem; }
  h2 { font-size: 1rem; margin-top: 1.75rem; }
  ol { padding-left: 1.2rem; }
  .warn { margin-top: 2rem; padding: .8rem 1rem; border-left: 3px solid currentColor; opacity: .8; font-size: .9rem; }
</style>
</head>
<body>
  <h1>Hi ${name} &mdash; here is your feed</h1>
  <p class="lede">This link is yours alone. Paste it into a podcast app once and new episodes arrive automatically.</p>

  <div class="feed">
    <input id="feed" value="${esc(feedUrl)}" readonly aria-label="Your private feed URL">
    <button id="copy" type="button">Copy</button>
  </div>

  <h2>iPhone / iPad</h2>
  <p>Apple Podcasts on iPhone cannot add a feed by URL. Use a free app that can:</p>
  <ol>
    <li>Install <strong>Overcast</strong> or <strong>Pocket Casts</strong> from the App Store.</li>
    <li>Overcast: <strong>+</strong> &rarr; <strong>Add URL</strong>. Pocket Casts: <strong>Discover</strong> &rarr; search icon &rarr; paste the URL.</li>
    <li>Paste the link above and confirm.</li>
  </ol>

  <h2>Mac</h2>
  <ol>
    <li>Open the <strong>Podcasts</strong> app.</li>
    <li>Menu bar: <strong>File &rarr; Add a Show by URL&hellip;</strong></li>
    <li>Paste the link above.</li>
  </ol>

  <h2>Android</h2>
  <ol>
    <li>Install <strong>Pocket Casts</strong> or <strong>AntennaPod</strong>.</li>
    <li>AntennaPod: <strong>Add Podcast</strong> &rarr; <strong>Add Podcast by RSS address</strong>.</li>
    <li>Paste the link above.</li>
  </ol>

  <p class="warn">Please don't forward this link. Each listener has a different one, so a shared link can be traced back and switched off on its own.</p>

<script>
  document.getElementById('copy').addEventListener('click', async () => {
    const field = document.getElementById('feed');
    const button = document.getElementById('copy');
    try {
      await navigator.clipboard.writeText(field.value);
    } catch {
      field.select();
      document.execCommand('copy');
    }
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = 'Copy'; }, 1800);
  });
</script>
</body>
</html>`;

  return new Response(html, {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'private, no-store',
      'X-Robots-Tag': 'noindex, nofollow',
    },
  });
}

/* ---------------------------------------------------------------- helpers */

function notFound() {
  return new Response('Not found', {
    status: 404,
    headers: { 'Cache-Control': 'no-store' },
  });
}

function encodePath(path) {
  return String(path || '')
    .split('/')
    .map(encodeURIComponent)
    .join('/');
}

function esc(value) {
  return String(value ?? '').replace(
    /[<>&'"]/g,
    (char) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', "'": '&apos;', '"': '&quot;' })[char],
  );
}

function rfc2822(value) {
  const date = new Date(value || Date.now());
  const safe = Number.isNaN(date.getTime()) ? new Date() : date;
  return safe.toUTCString().replace('GMT', '+0000');
}
