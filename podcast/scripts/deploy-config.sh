#!/usr/bin/env bash
#
# Copy configuration from the repo checkout onto the share the containers read.
#
# Run on node-a, after `git pull`:
#
#   /mnt/user/podcast/repo/podcast/scripts/deploy-config.sh
#
# WHY THIS EXISTS
#
# The orchestrator is started with PODCASTPIPE_CONFIG=/pipeline/config/show.yaml,
# which makes /mnt/user/podcast the config root. Clients therefore resolve to
# /mnt/user/podcast/clients/<name>/, NOT to the copy in the repo checkout at
# /mnt/user/podcast/repo/podcast/clients/<name>/.
#
# Those two directories look identical and are not the same files. Editing the
# repo copy, committing it, and pulling on node-a changes nothing that any
# container reads -- with no error, because the share copy is still perfectly
# valid config. A host name edit and a set of feed URL fixes were both lost this
# way, each time looking like the code had ignored the change.
#
# The split is not an accident and should not be collapsed by symlinking the
# whole tree: clients/<name>/assets/ holds the voice reference and the avatar
# footage, which are large binaries that belong on the share and not in git.
# So YAML is copied and assets are left strictly alone.
set -euo pipefail

REPO="${1:-/mnt/user/podcast/repo/podcast}"
SHARE="${2:-/mnt/user/podcast}"

if [[ ! -d "$REPO/clients" ]]; then
    echo "No clients/ under $REPO -- is that the repo checkout?" >&2
    echo "Usage: $0 [repo-podcast-dir] [share-dir]" >&2
    exit 1
fi
if [[ ! -d "$SHARE" ]]; then
    echo "Share directory $SHARE does not exist." >&2
    exit 1
fi

echo "repo:  $REPO"
echo "share: $SHARE"
echo

# find and cp rather than rsync: this has to run on an Unraid box and on any
# node, and cp is not something that can be missing.
#
# Copies YAML and nothing else, and never descends into assets/ -- so a voice
# reference or a base loop sitting beside the config is untouchable from here.
# Deletions are not propagated: dropping a client from the repo must not
# silently take its configuration, and its recordings, off the share.
STAMP="$(date +%Y%m%d-%H%M%S)"

# Every overwrite keeps a timestamped copy of what was there. The share config
# is hand-editable and people do edit it -- that is the whole reason this drift
# exists -- so a sync that can silently discard an edit made on the box is not
# something to run casually. With a backup beside it, it is.
copy_yaml() {
    local src="$1" dest="$2" changed=0
    while IFS= read -r -d '' file; do
        local rel="${file#"$src"/}"
        local target="$dest/$rel"
        if [[ -f "$target" ]] && cmp -s "$file" "$target"; then
            continue
        fi
        mkdir -p "$(dirname "$target")"
        if [[ -f "$target" ]]; then
            cp "$target" "$target.bak-$STAMP"
            cp "$file" "$target"
            echo "  updated $rel  (previous kept as $(basename "$target").bak-$STAMP)"
        else
            cp "$file" "$target"
            echo "  added   $rel"
        fi
        changed=$((changed + 1))
    done < <(find "$src" -type d -name assets -prune -o -type f -name '*.yaml' -print0)
    if [[ $changed -eq 0 ]]; then
        echo "  (up to date)"
    fi
}

echo "clients:"
copy_yaml "$REPO/clients" "$SHARE/clients"

# cluster.yaml describes node URLs and roles, and is genuinely owned by the repo.
#
# config/show.yaml and config/sources.yaml are different: they are the
# single-tenant defaults, the repo ships them as templates, and the copy on the
# share is live configuration that predates the client profiles. Overwriting
# those from a template would be a regression, not a deploy, so they are only
# placed when absent.
echo
echo "config:"
mkdir -p "$SHARE/config"
if [[ -f "$REPO/config/cluster.yaml" ]]; then
    copy_yaml_file() {
        local file="$1" target="$2"
        if [[ -f "$target" ]] && cmp -s "$file" "$target"; then
            echo "  (up to date) $(basename "$target")"
        elif [[ -f "$target" ]]; then
            cp "$target" "$target.bak-$STAMP"
            cp "$file" "$target"
            echo "  updated $(basename "$target")  (previous kept as $(basename "$target").bak-$STAMP)"
        else
            cp "$file" "$target"
            echo "  added   $(basename "$target")"
        fi
    }
    copy_yaml_file "$REPO/config/cluster.yaml" "$SHARE/config/cluster.yaml"
fi
for name in show.yaml sources.yaml; do
    if [[ -f "$REPO/config/$name" && ! -f "$SHARE/config/$name" ]]; then
        cp "$REPO/config/$name" "$SHARE/config/$name"
        echo "  added   $name (template — the share had none)"
    fi
done

echo
echo "Done."
echo "Restart the orchestrator to pick up code changes from the same pull:"
echo "  docker restart podcast-orchestrator"
