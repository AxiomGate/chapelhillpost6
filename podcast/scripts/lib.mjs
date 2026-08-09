import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const PODCAST_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const WORKER_DIR = resolve(PODCAST_DIR, 'worker');
export const CONFIG_PATH = resolve(PODCAST_DIR, 'podcast.config.json');

export function loadConfig() {
  if (!existsSync(CONFIG_PATH)) {
    fail(
      `No config found at ${CONFIG_PATH}\n` +
        `Copy podcast.config.example.json to podcast.config.json and fill it in.`,
    );
  }
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
  } catch (error) {
    fail(`Could not parse ${CONFIG_PATH}: ${error.message}`);
  }
}

/**
 * Wrangler renamed the KV subcommands between v3 (`kv:key`) and v4 (`kv key`),
 * and v4 made remote operations opt-in with `--remote`. Detect once and adapt
 * rather than pinning the user to one version.
 */
let cachedWrangler;
export function wranglerDialect() {
  if (cachedWrangler) return cachedWrangler;

  const result = spawnSync(wranglerBin(), ['--version'], {
    encoding: 'utf8',
    cwd: WORKER_DIR,
    shell: process.platform === 'win32',
  });
  if (result.error) {
    fail(
      'Could not run wrangler. Install it with:\n  npm install -g wrangler\n' +
        `Underlying error: ${result.error.message}`,
    );
  }

  const text = `${result.stdout || ''}${result.stderr || ''}`;
  const match = /(\d+)\.\d+\.\d+/.exec(text);
  const major = match ? Number.parseInt(match[1], 10) : 4;

  cachedWrangler = {
    major,
    kv: major >= 4 ? ['kv', 'key'] : ['kv:key'],
    remote: major >= 4 ? ['--remote'] : [],
  };
  return cachedWrangler;
}

export function wranglerBin() {
  return process.env.WRANGLER_BIN || 'wrangler';
}

export function wrangler(args, { capture = true, allowFailure = false } = {}) {
  const result = spawnSync(wranglerBin(), args, {
    cwd: WORKER_DIR,
    encoding: 'utf8',
    stdio: capture ? 'pipe' : 'inherit',
    shell: process.platform === 'win32',
  });

  if (result.error) fail(`wrangler failed to start: ${result.error.message}`);
  if (result.status !== 0 && !allowFailure) {
    const detail = capture ? `\n${result.stderr || result.stdout || ''}`.trimEnd() : '';
    fail(`wrangler ${args.join(' ')} exited with code ${result.status}${detail}`);
  }
  return result;
}

export function kvArgs(config, verb, extra = []) {
  const { kv, remote } = wranglerDialect();
  return [...kv, verb, ...extra, `--binding=${config.kvBinding || 'SUBSCRIBERS'}`, ...remote];
}

/**
 * wrangler prints human-readable preamble lines before JSON output, so scan
 * for the first line that actually starts a JSON document.
 */
export function parseJsonOutput(text) {
  const trimmed = (text || '').trim();
  if (!trimmed) return null;

  const start = trimmed.search(/[[{]/);
  if (start === -1) return null;

  try {
    return JSON.parse(trimmed.slice(start));
  } catch {
    return null;
  }
}

export function contentTypeFor(filename) {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
  return (
    {
      '.mp3': 'audio/mpeg',
      '.m4a': 'audio/x-m4a',
      '.m4b': 'audio/x-m4b',
      '.aac': 'audio/aac',
      '.wav': 'audio/wav',
      '.ogg': 'audio/ogg',
      '.opus': 'audio/opus',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.json': 'application/json',
    }[ext] || 'application/octet-stream'
  );
}

export function fail(message) {
  console.error(`\n${message}\n`);
  process.exit(1);
}
