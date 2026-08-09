#!/usr/bin/env node
/**
 * Manage the people who can reach the feed.
 *
 *   node scripts/subscribers.mjs add "Dave K" dave@example.com
 *   node scripts/subscribers.mjs list
 *   node scripts/subscribers.mjs revoke "Dave K"
 *   node scripts/subscribers.mjs revoke <token>
 *
 * Tokens live only in Cloudflare KV. Nothing here writes them to disk or to
 * the repository, so a public repo stays safe to publish.
 */

import { randomBytes } from 'node:crypto';
import { fail, kvArgs, loadConfig, parseJsonOutput, wrangler } from './lib.mjs';

const config = loadConfig();
const [command, ...args] = process.argv.slice(2);

switch (command) {
  case 'add':
    await add(args[0], args[1]);
    break;
  case 'list':
    await list();
    break;
  case 'revoke':
    await revoke(args[0]);
    break;
  default:
    usage();
}

function usage() {
  console.log(`
Usage:
  node scripts/subscribers.mjs add "<name>" [email]
  node scripts/subscribers.mjs list
  node scripts/subscribers.mjs revoke <name|token>
`);
  process.exit(command ? 1 : 0);
}

async function add(name, email) {
  if (!name) fail('A name is required: subscribers.mjs add "Dave K" dave@example.com');

  const existing = await fetchAll();
  if (existing.some((entry) => sameName(entry.name, name))) {
    fail(`"${name}" already has a feed. Revoke it first if you want a fresh link.`);
  }

  // 16 bytes of randomness, base64url-encoded: 22 characters, not guessable
  // and not brute-forceable at any rate a Worker will ever serve.
  const token = randomBytes(16).toString('base64url');
  const record = {
    name,
    email: email || '',
    createdAt: new Date().toISOString(),
    lastSeen: '',
    hits: 0,
  };

  wrangler(kvArgs(config, 'put', [`sub:${token}`, JSON.stringify(record)]));

  const origin = config.feedOrigin.replace(/\/$/, '');
  console.log(`
Added ${name}.

Send them this link — it shows setup instructions for their app:
  ${origin}/f/${token}

The raw feed URL, if they ask for it directly:
  ${origin}/f/${token}.xml
`);
}

async function list() {
  const entries = await fetchAll();
  if (entries.length === 0) {
    console.log('\nNo subscribers yet. Add one with: subscribers.mjs add "Name"\n');
    return;
  }

  const origin = config.feedOrigin.replace(/\/$/, '');
  console.log('');
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const seen = entry.lastSeen
      ? `${new Date(entry.lastSeen).toLocaleString()} (${entry.hits} check-ins)`
      : 'never';
    console.log(`  ${entry.name}${entry.email ? ` <${entry.email}>` : ''}`);
    console.log(`    link       ${origin}/f/${entry.token}`);
    console.log(`    added      ${new Date(entry.createdAt).toLocaleDateString()}`);
    console.log(`    last seen  ${seen}`);
    console.log('');
  }
}

async function revoke(target) {
  if (!target) fail('Give a name or a token: subscribers.mjs revoke "Dave K"');

  const entries = await fetchAll();
  const match =
    entries.find((entry) => entry.token === target) ||
    entries.find((entry) => sameName(entry.name, target));

  if (!match) fail(`No subscriber matches "${target}". Run "list" to see them all.`);

  wrangler(kvArgs(config, 'delete', [`sub:${match.token}`]), { capture: false });
  console.log(`
Revoked ${match.name}. Their link stops working within about a minute
(KV changes take a moment to reach every edge location).
`);
}

async function fetchAll() {
  const result = wrangler(kvArgs(config, 'list'));
  const keys = parseJsonOutput(result.stdout);
  if (!Array.isArray(keys)) return [];

  const entries = [];
  for (const key of keys) {
    if (!key.name?.startsWith('sub:')) continue;

    const token = key.name.slice(4);
    const raw = wrangler(kvArgs(config, 'get', [key.name]), { allowFailure: true });
    const record = parseJsonOutput(raw.stdout) || {};
    entries.push({ token, name: token, email: '', createdAt: '', lastSeen: '', hits: 0, ...record });
  }
  return entries;
}

function sameName(a, b) {
  return String(a).trim().toLowerCase() === String(b).trim().toLowerCase();
}
