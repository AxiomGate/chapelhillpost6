#!/usr/bin/env node
/**
 * Publish new episodes.
 *
 *   node scripts/publish.mjs              # upload anything new, update the feed
 *   node scripts/publish.mjs --dry-run    # show what would happen, change nothing
 *
 * Reads the audio folder named in podcast.config.json (your NAS share), uploads
 * files R2 has not seen yet, and rewrites the feed metadata. Files already
 * published are skipped, so running it twice is harmless.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { basename, extname, join, resolve } from 'node:path';
import {
  contentTypeFor,
  fail,
  loadConfig,
  PODCAST_DIR,
  wrangler,
  wranglerDialect,
} from './lib.mjs';

const AUDIO_EXTENSIONS = new Set(['.mp3', '.m4a', '.m4b', '.aac', '.wav', '.ogg', '.opus']);
const METADATA_KEY = 'episodes.json';

const config = loadConfig();
const dryRun = process.argv.includes('--dry-run');
const localDir = resolve(PODCAST_DIR, 'local');
const metadataPath = join(localDir, METADATA_KEY);

if (!config.sourceDir) fail('podcast.config.json needs a "sourceDir".');
if (!existsSync(config.sourceDir)) {
  fail(
    `Cannot reach ${config.sourceDir}\n` +
      `Check the NAS is online and the share is mounted. On Windows you can also\n` +
      `map it to a drive letter and use e.g. "P:\\\\" as sourceDir.`,
  );
}

mkdirSync(localDir, { recursive: true });

const data = pullMetadata();
data.show = { ...data.show, ...config.show };

const published = new Set((data.episodes || []).map((episode) => basename(episode.file)));
const candidates = readdirSync(config.sourceDir)
  .filter((name) => AUDIO_EXTENSIONS.has(extname(name).toLowerCase()))
  .filter((name) => !published.has(name))
  .sort();

if (candidates.length === 0) {
  console.log('\nNothing new to publish. The feed is already up to date.\n');
  process.exit(0);
}

console.log(`\nFound ${candidates.length} new file(s) in ${config.sourceDir}:\n`);

const added = [];
for (const name of candidates) {
  const fullPath = join(config.sourceDir, name);
  const stats = statSync(fullPath);
  const key = `audio/${name}`;
  const episode = {
    id: name.replace(extname(name), ''),
    title: titleFrom(name),
    description: '',
    file: key,
    size: stats.size,
    type: contentTypeFor(name),
    duration: probeDuration(fullPath),
    pubDate: pubDateFrom(name, stats),
  };

  console.log(`  ${name}`);
  console.log(`    title    ${episode.title}`);
  console.log(`    date     ${new Date(episode.pubDate).toLocaleString()}`);
  console.log(`    size     ${(stats.size / 1048576).toFixed(1)} MB`);
  if (episode.duration) console.log(`    duration ${episode.duration}`);

  if (!dryRun) {
    r2Put(key, fullPath, episode.type);
  }
  added.push(episode);
  console.log('');
}

if (config.coverArtFile && existsSync(config.coverArtFile)) {
  const coverKey = `cover${extname(config.coverArtFile).toLowerCase()}`;
  data.show.coverArt = coverKey;
  if (!dryRun) r2Put(coverKey, config.coverArtFile, contentTypeFor(config.coverArtFile));
}

data.episodes = [...(data.episodes || []), ...added];

if (dryRun) {
  console.log('Dry run — nothing was uploaded.\n');
  process.exit(0);
}

writeFileSync(metadataPath, `${JSON.stringify(data, null, 2)}\n`);
r2Put(METADATA_KEY, metadataPath, 'application/json');

console.log(
  `Published ${added.length} episode(s). ${data.episodes.length} total in the feed.\n` +
    `Subscribers pick them up on their app's next refresh.\n`,
);

/* ------------------------------------------------------------------ r2 io */

function r2Put(key, file, contentType) {
  const { remote } = wranglerDialect();
  wrangler([
    'r2',
    'object',
    'put',
    `${config.bucket}/${key}`,
    `--file=${file}`,
    `--content-type=${contentType}`,
    ...remote,
  ]);
}

function pullMetadata() {
  const { remote } = wranglerDialect();
  const result = wrangler(
    ['r2', 'object', 'get', `${config.bucket}/${METADATA_KEY}`, `--file=${metadataPath}`, ...remote],
    { allowFailure: true },
  );

  // A missing object is the normal first-run case, not an error.
  if (result.status !== 0 || !existsSync(metadataPath)) {
    return { show: { ...config.show }, episodes: [] };
  }

  try {
    return JSON.parse(readFileSync(metadataPath, 'utf8'));
  } catch {
    fail(
      `The episodes.json in R2 is not valid JSON. Fix or delete it before publishing:\n` +
        `  wrangler r2 object delete ${config.bucket}/${METADATA_KEY}`,
    );
  }
}

/* ---------------------------------------------------------------- helpers */

/** Turn "2026-08-09-morning-report.mp3" into "Morning Report". */
function titleFrom(filename) {
  const stem = filename
    .replace(extname(filename), '')
    .replace(/^\d{4}[-_]\d{2}[-_]\d{2}[-_\s]*/, '')
    .replace(/[-_]+/g, ' ')
    .trim();

  if (!stem) return filename.replace(extname(filename), '');
  return stem.replace(/\b\w/g, (char) => char.toUpperCase());
}

/** Prefer a date in the filename; fall back to the file's modified time. */
function pubDateFrom(filename, stats) {
  const match = /^(\d{4})[-_](\d{2})[-_](\d{2})/.exec(filename);
  if (match) {
    const [, year, month, day] = match;
    const hour = config.publishHourUtc ?? 11;
    return new Date(Date.UTC(+year, +month - 1, +day, hour, 0, 0)).toISOString();
  }
  return stats.mtime.toISOString();
}

/** Optional: ffprobe gives a real duration. Without it the tag is omitted. */
function probeDuration(file) {
  const result = spawnSync(
    'ffprobe',
    ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file],
    { encoding: 'utf8', shell: process.platform === 'win32' },
  );
  if (result.error || result.status !== 0) return '';

  const seconds = Math.round(Number.parseFloat(result.stdout));
  if (!Number.isFinite(seconds) || seconds <= 0) return '';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  const pad = (value) => String(value).padStart(2, '0');
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(rest)}` : `${minutes}:${pad(rest)}`;
}
